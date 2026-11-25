use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};

use drm::control::atomic::AtomicModeReq;
use drm::control::{
    connector, crtc, framebuffer, plane, property, AtomicCommitFlags, Device as ControlDevice,
    RawResourceHandle, ResourceHandle,
};

use super::DrmDeviceFd;
use crate::backend::drm::error::AccessError;
use crate::{backend::drm::error::Error, utils::DevPath};

use tracing::{debug, error, info_span, trace};

/// One property of one object as it was before we took the device over.
#[derive(Clone, Debug)]
struct CapturedProp {
    name: String,
    property: property::Handle,
    value: property::RawValue,
}

/// The properties of one object (a connector, CRTC or plane) that can be
/// replayed in the exit restore.
#[derive(Clone, Debug)]
struct CapturedObject {
    object: RawResourceHandle,
    props: Vec<CapturedProp>,
}

/// One CRTC as it was: its restorable properties, and its mode.
#[derive(Clone, Debug)]
struct CapturedCrtc {
    object: CapturedObject,
    /// The CRTC's `MODE_ID` property, and a mode blob of our own holding the
    /// mode the CRTC was scanning out at creation (`0` if it was scanning out
    /// nothing). `None` when no restorable mode could be captured, in which
    /// case the CRTC is not restored at exit at all.
    ///
    /// The raw `MODE_ID` value is deliberately not kept: it names a blob the
    /// kernel can free as soon as our own modeset replaces the mode, and a
    /// dead id would make the whole restore commit fail with `EINVAL`.
    mode: Option<(property::Handle, property::RawValue)>,
}

/// The device state to restore when the compositor exits, captured at
/// creation, before anything resets the device.
///
/// Captured for the restore commit only what the atomic uAPI can accept
/// back: DPMS, immutable properties, fence properties and framebuffer
/// objects are structurally refused by the kernel and never captured,
/// and blob ids that can go stale while we run are captured as copies of
/// our own, which this fd keeps alive until the restore has been
/// committed. (The legacy device does the same job with `set_crtc`; this
/// is its atomic counterpart.)
#[derive(Debug, Default)]
struct PreviousState {
    connectors: Vec<CapturedObject>,
    crtcs: Vec<CapturedCrtc>,
    planes: Vec<CapturedObject>,
}

#[derive(Clone, Debug, Default)]
pub struct PropMapping {
    pub connectors: HashMap<connector::Handle, HashMap<String, property::Handle>>,
    pub crtcs: HashMap<crtc::Handle, HashMap<String, property::Handle>>,
    pub planes: HashMap<plane::Handle, HashMap<String, property::Handle>>,
}

impl PropMapping {
    pub(crate) fn conn_prop_handle(
        &self,
        handle: connector::Handle,
        name: &'static str,
    ) -> Result<property::Handle, Error> {
        self.connectors
            .get(&handle)
            .ok_or(Error::UnknownConnector(handle))?
            .get(name)
            .ok_or_else(|| Error::UnknownProperty {
                handle: handle.into(),
                name,
            })
            .copied()
    }

    pub(crate) fn crtc_prop_handle(
        &self,
        handle: crtc::Handle,
        name: &'static str,
    ) -> Result<property::Handle, Error> {
        self.crtcs
            .get(&handle)
            .ok_or(Error::UnknownCrtc(handle))?
            .get(name)
            .ok_or_else(|| Error::UnknownProperty {
                handle: handle.into(),
                name,
            })
            .copied()
    }

    pub(crate) fn plane_prop_handle(
        &self,
        handle: plane::Handle,
        name: &'static str,
    ) -> Result<property::Handle, Error> {
        self.planes
            .get(&handle)
            .ok_or(Error::UnknownPlane(handle))?
            .get(name)
            .ok_or_else(|| Error::UnknownProperty {
                handle: handle.into(),
                name,
            })
            .copied()
    }
}

#[derive(Debug)]
pub struct AtomicDrmDevice {
    pub(crate) fd: DrmDeviceFd,
    pub(crate) active: Arc<AtomicBool>,
    old_state: PreviousState,
    pub(crate) prop_mapping: Arc<RwLock<PropMapping>>,
    pub(super) span: tracing::Span,
}

impl AtomicDrmDevice {
    pub fn new(fd: DrmDeviceFd, active: Arc<AtomicBool>, disable_connectors: bool) -> Result<Self, Error> {
        let span = info_span!("drm_atomic");
        let mut dev = AtomicDrmDevice {
            fd,
            active,
            old_state: Default::default(),
            prop_mapping: Default::default(),
            span,
        };
        let _guard = dev.span.enter();

        // Enumerate (and save) the current device state.
        let res_handles = dev.fd.resource_handles().map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading drm resources",
                dev: dev.fd.dev_path(),
                source,
            })
        })?;

        let planes = dev.fd.plane_handles().map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading planes",
                dev: dev.fd.dev_path(),
                source,
            })
        })?;

        // Capture what the exit restore will commit, before the reset below
        // changes the device. See `PreviousState` for what is kept and why.
        dev.old_state =
            PreviousState::capture(&dev.fd, res_handles.connectors(), res_handles.crtcs(), &planes)?;

        // And because the mapping is not consistent across devices,
        // we also need to lookup the handle for a property name.
        // And we do this a fair bit, so lets cache that mapping.
        let mut mapping = dev.prop_mapping.write().unwrap();
        map_props(&dev.fd, res_handles.connectors(), &mut mapping.connectors)?;
        map_props(&dev.fd, res_handles.crtcs(), &mut mapping.crtcs)?;
        map_props(&dev.fd, &planes, &mut mapping.planes)?;

        trace!("Mapping: {:#?}", mapping);

        drop(mapping);

        // If the user does not explicitly requests us to skip this,
        // we clear out the complete connector<->crtc mapping on device creation.
        //
        // The reason is, that certain operations may be racy otherwise. Surfaces can
        // exist on different threads: as a result, we cannot really enumerate the current state
        // (it might be changed on another thread during the enumeration). And commits can fail,
        // if e.g. a connector is already bound to another surface, which is difficult to analyse at runtime.
        //
        // An easy workaround is to set a known state on device creation, so we can only
        // run into these errors on our own and not because previous compositors left the device
        // in a funny state.
        if disable_connectors {
            debug!("Resetting drm device to known state");
            dev.reset_state()?;
        }

        drop(_guard);
        Ok(dev)
    }

    pub(super) fn reset_state(&self) -> Result<(), Error> {
        // reset state sets the connectors into a known state (all disabled),
        // for the same reasons we do this on device creation.
        //
        // We might end up with conflicting commit requirements, if we want to restore our state,
        // on top of the state the previous compositor left the device in.
        // This is because we do commits per surface and not per device, so we do a global
        // commit here, to fix any conflicts.
        let res_handles = self.fd.resource_handles().map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading drm resources",
                dev: self.fd.dev_path(),
                source,
            })
        })?;
        let plane_handles = self.fd.plane_handles().map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading drm plane resources",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        let mut prop_mapping = self.prop_mapping.write().unwrap();

        // Make sure the mapping is up to date
        prop_mapping.connectors.clear();
        map_props(&self.fd, res_handles.connectors(), &mut prop_mapping.connectors)?;

        // Disable all connectors (otherwise we might run into conflicting commits when restarting the rendering loop)
        let mut req = AtomicModeReq::new();
        for conn in res_handles.connectors() {
            let prop = prop_mapping
                .conn_prop_handle(*conn, "CRTC_ID")
                .expect("Unknown property CRTC_ID");
            req.add_property(*conn, prop, property::Value::CRTC(None));
        }
        // Disable all planes
        for plane in plane_handles {
            let prop = prop_mapping
                .plane_prop_handle(plane, "CRTC_ID")
                .expect("Unknown property CRTC_ID");
            req.add_property(plane, prop, property::Value::CRTC(None));

            let prop = prop_mapping
                .plane_prop_handle(plane, "FB_ID")
                .expect("Unknown property FB_ID");
            req.add_property(plane, prop, property::Value::Framebuffer(None));
        }
        // A crtc without a connector has no mode, we also need to reset that.
        // Otherwise the commit will not be accepted.
        for crtc in res_handles.crtcs() {
            let mode_prop = prop_mapping
                .crtc_prop_handle(*crtc, "MODE_ID")
                .expect("Unknown property MODE_ID");
            let active_prop = prop_mapping
                .crtc_prop_handle(*crtc, "ACTIVE")
                .expect("Unknown property ACTIVE");
            req.add_property(*crtc, active_prop, property::Value::Boolean(false));
            req.add_property(*crtc, mode_prop, property::Value::Unknown(0));
        }
        self.fd
            .atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to disable connectors",
                    dev: self.fd.dev_path(),
                    source,
                })
            })?;

        Ok(())
    }
}

impl AtomicDrmDevice {
    /// Put the card back the way it was found.
    ///
    /// Called from `Drop` for every path that lets the device go, and directly
    /// by a compositor whose teardown knows the session is ending: a device
    /// that outlives its compositor — reached from a detached thread, or from
    /// a handle another object still holds — never runs `Drop` before the
    /// process calls `_exit`, and then the console is left as the compositor
    /// left it. Doing it explicitly needs no ownership the `Arc` does not
    /// give, which is why this takes `&self`: the captured state is read-only.
    pub(crate) fn restore_previous_state(&self) {
        let _guard = self.span.enter();

        // Here we restore the card to the state it was in when we took it
        // over: the modes and plane framebuffers that e.g. a getty was
        // scanning out go back on, so its framebuffer is visible again the
        // moment we exit. The legacy device does the same with `set_crtc`;
        // this is its atomic counterpart.
        //
        // What the commit carries was decided at capture time (see
        // `PreviousState`): every property the atomic uAPI can accept back,
        // with blob ids that our own modesets could have invalidated
        // committed as copies we hold instead. What the uAPI structurally
        // refuses — DPMS, immutable properties, fence properties, the
        // framebuffers themselves — is not in the commit at all.
        //
        // What cannot be restored is left as `clear_state` left it, off:
        // a CRTC whose capture failed or whose connectors went away while
        // we ran, and a plane whose captured framebuffer no longer exists
        // (a previous compositor's — its fd close destroyed it). Pointing
        // a commit at any of those would fail the whole commit, and one
        // CRTC that cannot come back must not take the others with it.
        //
        // A remaining failure is real and is logged: the screens stay as
        // they are until the console repaints, which it does on its next
        // setcrtc.

        debug!("Device still active, trying to restore previous state");

        let gone_crtcs = self.old_state.unrestorable_crtcs(&self.fd);
        let dead_planes = self.old_state.planes_with_dead_framebuffers(&self.fd);

        let entries = restore_entries(&self.old_state, &gone_crtcs, &dead_planes);
        let mut req = AtomicModeReq::new();
        for entry in &entries {
            req.add_raw_property(entry.object, entry.property, entry.value);
        }
        trace!("Previous state: {:?}", entries);
        if let Err(err) = self.fd.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req) {
            error!("Failed to restore previous state. Error: {}", err);
        }
    }
}

impl Drop for AtomicDrmDevice {
    fn drop(&mut self) {
        if !self.active.load(Ordering::SeqCst) {
            return;
        }
        self.restore_previous_state();
    }
}

// Capture the state the exit restore will commit.
//
// The whole point of this module's restore is that the request it builds is
// one the kernel actually accepts, so every decision about what that
// request may carry is made here, against the kernel's own rules rather
// than against a list of property names we would have to keep current
// driver by driver: what the atomic uAPI structurally cannot accept is not
// captured at all, and what it accepts but whose raw value can go stale is
// captured in a form that cannot.
impl PreviousState {
    fn capture<D>(
        fd: &D,
        connectors: &[connector::Handle],
        crtcs: &[crtc::Handle],
        planes: &[plane::Handle],
    ) -> Result<Self, Error>
    where
        D: DevPath + ControlDevice,
    {
        let mut state = PreviousState::default();

        for &conn in connectors {
            state.connectors.push(capture_object(fd, conn)?);
        }

        for &crtc_handle in crtcs {
            let object = capture_object(fd, crtc_handle)?;

            // The mode is read through the legacy CRTC ioctl and kept as a
            // blob of our own. `MODE_ID`'s raw value names a blob the kernel
            // can free as soon as our own modeset replaces that mode, and a
            // dead blob id would fail the whole restore commit.
            let info = fd.get_crtc(crtc_handle).map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Error loading crtc info",
                    dev: fd.dev_path(),
                    source,
                })
            })?;
            let mode = match object.props.iter().find(|prop| prop.name == "MODE_ID") {
                // A CRTC without a MODE_ID property cannot be restored this
                // way at all; it is skipped at exit rather than half-set.
                None => None,
                Some(mode_prop) => {
                    let blob = match info.mode() {
                        Some(mode) => {
                            let blob = fd.create_property_blob(&mode).map_err(|source| {
                                Error::Access(AccessError {
                                    errmsg: "Failed to create mode blob for the exit restore",
                                    dev: fd.dev_path(),
                                    source,
                                })
                            })?;
                            let property::Value::Blob(blob) = blob else {
                                unreachable!("create_property_blob answers with a blob id")
                            };
                            blob
                        }
                        // The CRTC was scanning out nothing; restoring it
                        // means MODE_ID 0, which stays valid forever.
                        None => 0,
                    };
                    Some((mode_prop.property, blob))
                }
            };

            state.crtcs.push(CapturedCrtc { object, mode });
        }

        for &plane in planes {
            state.planes.push(capture_object(fd, plane)?);
        }

        Ok(state)
    }

    /// The CRTC a captured connector was bound to, if any.
    fn captured_binding(connector: &CapturedObject) -> Option<crtc::Handle> {
        let prop = connector.props.iter().find(|prop| prop.name == "CRTC_ID")?;
        RawResourceHandle::new(prop.value as u32).map(crtc::Handle::from)
    }

    /// CRTCs that cannot be put back at exit: those whose capture failed,
    /// or on which every connector captured at creation has gone away.
    fn unrestorable_crtcs<D>(&self, fd: &D) -> Vec<crtc::Handle>
    where
        D: DevPath + ControlDevice,
    {
        let mut gone: Vec<crtc::Handle> = self
            .crtcs
            .iter()
            .filter(|crtc| crtc.mode.is_none())
            .map(|crtc| crtc::Handle::from(crtc.object.object))
            .collect();

        // A CRTC is only restorable if at least one connector captured on
        // it is still there. (The TODO in the old restore: replaying a
        // connector that was unplugged mid-session would have failed the
        // whole commit, taking every other output down with it.)
        let mut bindings: HashMap<crtc::Handle, Vec<(connector::Handle, bool)>> = HashMap::new();
        for conn in &self.connectors {
            let Some(bound) = Self::captured_binding(conn) else {
                continue;
            };
            let handle = connector::Handle::from(conn.object);
            // A cached probe, not a forced one: a forced probe at exit can
            // flicker, and the kernel's status is what the next compositor
            // will be given anyway.
            let connected = match fd.get_connector(handle, false) {
                Ok(info) => info.state() != connector::State::Disconnected,
                // The connector object itself is gone.
                Err(_) => false,
            };
            bindings.entry(bound).or_default().push((handle, connected));
        }
        for (bound, votes) in bindings {
            if !votes.is_empty() && votes.iter().all(|&(_, connected)| !connected) {
                debug!(
                    crtc = ?bound,
                    connectors = ?votes.iter().map(|&(conn, _)| conn).collect::<Vec<_>>(),
                    "not restoring a crtc whose connectors are gone"
                );
                gone.push(bound);
            }
        }
        gone
    }

    /// Planes whose captured framebuffer no longer exists. The fbcon
    /// framebuffer the getty uses is made by the kernel and stays alive;
    /// one that does not is a previous compositor's, destroyed when its fd
    /// closed, and replaying its id would fail the whole restore commit.
    fn planes_with_dead_framebuffers<D>(&self, fd: &D) -> Vec<plane::Handle>
    where
        D: DevPath + ControlDevice,
    {
        let mut dead = Vec::new();
        for plane in &self.planes {
            let Some(fb) = plane
                .props
                .iter()
                .find(|prop| prop.name == "FB_ID")
                .filter(|prop| prop.value != 0)
            else {
                continue;
            };
            let fb = framebuffer::Handle::from(
                RawResourceHandle::new(fb.value as u32).expect("nonzero checked above"),
            );
            if fd.get_framebuffer(fb).is_err() {
                let handle = plane::Handle::from(plane.object);
                debug!(plane = ?handle, fb = ?fb, "not restoring a plane whose framebuffer is gone");
                dead.push(handle);
            }
        }
        dead
    }
}

/// Capture the restorable properties of one object.
fn capture_object<D, T>(fd: &D, handle: T) -> Result<CapturedObject, Error>
where
    D: DevPath + ControlDevice,
    T: ResourceHandle,
{
    let props = fd.get_properties(handle).map_err(|source| {
        Error::Access(AccessError {
            errmsg: "Error reading properties",
            dev: fd.dev_path(),
            source,
        })
    })?;

    let mut object = CapturedObject {
        object: handle.into(),
        props: Vec::with_capacity(props.as_props_and_values().0.len()),
    };
    for (prop, &value) in props.iter() {
        let info = fd.get_property(*prop).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error reading property info",
                dev: fd.dev_path(),
                source,
            })
        })?;
        let name = info.name().to_string_lossy();
        let action = capture_action(
            &name,
            info.mutable(),
            info.atomic(),
            matches!(info.value_type(), property::ValueType::Blob),
        );
        match action {
            CaptureAction::Skip => {}
            CaptureAction::Raw => object.props.push(CapturedProp {
                name: name.into_owned(),
                property: *prop,
                value,
            }),
            CaptureAction::CopyBlob => {
                // 0 is "no blob" and stays valid; anything else is copied,
                // so what the restore commits is an id this fd keeps alive.
                let value = if value == 0 { value } else { copy_blob(fd, value)? };
                object.props.push(CapturedProp {
                    name: name.into_owned(),
                    property: *prop,
                    value,
                });
            }
        }
    }
    Ok(object)
}

/// Copy a property blob, as an id of our own.
///
/// The kernel frees a blob once the last holder lets go, and our own
/// modesets release the holders the pre-session state had: the gamma LUT a
/// night-light client set through the legacy ioctl, the connector's HDR
/// metadata, a replaced mode. A captured id can therefore be dead by the
/// time it is replayed, and one dead id fails the whole restore commit.
/// A copy made now is kept alive by this fd until the restore has been
/// committed, so it cannot be.
fn copy_blob<D>(fd: &D, blob: property::RawValue) -> Result<property::RawValue, Error>
where
    D: DevPath + ControlDevice,
{
    let mut data = fd.get_property_blob(blob).map_err(|source| {
        Error::Access(AccessError {
            errmsg: "Failed to read a property blob for the exit restore",
            dev: fd.dev_path(),
            source,
        })
    })?;
    let copy = drm_ffi::mode::create_property_blob(fd.as_fd(), &mut data).map_err(|source| {
        Error::Access(AccessError {
            errmsg: "Failed to copy a property blob for the exit restore",
            dev: fd.dev_path(),
            source,
        })
    })?;
    Ok(copy.blob_id as property::RawValue)
}

/// What one enumerated property contributes to the exit restore.
///
/// This is the kernel's rulebook for what an atomic commit can accept,
/// looked up per property at capture time:
///
/// - DPMS is refused outright by `drm_atomic_connector_set_property`
///   ("legacy ... can only be set via legacy uAPI"); `ACTIVE`, which is
///   captured, is its atomic counterpart.
/// - Immutable properties — a connector's EDID, a plane's `type` and
///   IN_FORMATS — are refused by the property validator.
/// - The ATOMIC flag marks properties whose values are not replayable
///   state at all: IN_FENCE_FD and friends carry file descriptors and
///   addresses, in both directions.
/// - MODE_ID's blob id is invalidated by our own modesets; the mode itself
///   is captured through the legacy CRTC ioctl and re-created as a blob
///   of our own.
///
/// Everything else mutable is replayed as captured, blob properties as
/// copies (see [`copy_blob`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureAction {
    /// Not restorable through the atomic uAPI; left out of the commit.
    Skip,
    /// Replayed with the value as captured.
    Raw,
    /// Replayed with a copy made at capture time, which our fd keeps
    /// alive.
    CopyBlob,
}

fn capture_action(name: &str, mutable: bool, atomic_only: bool, is_blob: bool) -> CaptureAction {
    if name == "DPMS" {
        return CaptureAction::Skip;
    }
    if atomic_only {
        return CaptureAction::Skip;
    }
    if !mutable {
        return CaptureAction::Skip;
    }
    if name == "MODE_ID" {
        return CaptureAction::Skip;
    }
    if is_blob {
        CaptureAction::CopyBlob
    } else {
        CaptureAction::Raw
    }
}

/// One (object, property, value) entry of the restore commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RestoreEntry {
    object: RawResourceHandle,
    property: property::Handle,
    value: property::RawValue,
}

/// Build the entries of the restore commit from the captured state.
///
/// Pure on purpose — connector status, framebuffer liveness and blob
/// copies are ioctls and happen before this — so the composition below
/// can be tested without hardware.
///
/// `gone_crtcs` are CRTCs that cannot be put back. Their connectors and
/// themselves are left out entirely (they are off since `clear_state`, and
/// a commit pointing a connector at a CRTC it does not enable is refused
/// outright), and the planes captured on them are cleared instead of
/// replayed, for the same reason. `dead_fb_planes` are planes whose
/// captured framebuffer no longer exists; cleared likewise.
fn restore_entries(
    state: &PreviousState,
    gone_crtcs: &[crtc::Handle],
    dead_fb_planes: &[plane::Handle],
) -> Vec<RestoreEntry> {
    let mut entries = Vec::new();

    for conn in &state.connectors {
        if let Some(bound) = PreviousState::captured_binding(conn) {
            if gone_crtcs.contains(&bound) {
                continue;
            }
        }
        push_props(&mut entries, conn, &[]);
    }

    for crtc in &state.crtcs {
        let handle = crtc::Handle::from(crtc.object.object);
        if gone_crtcs.contains(&handle) {
            continue;
        }
        push_props(&mut entries, &crtc.object, &[]);
        if let Some((prop, blob)) = crtc.mode {
            entries.push(RestoreEntry {
                object: crtc.object.object,
                property: prop,
                value: blob,
            });
        }
    }

    for plane in &state.planes {
        let handle = plane::Handle::from(plane.object);
        let bound = PreviousState::captured_binding(plane);
        let clear = dead_fb_planes.contains(&handle) || bound.is_some_and(|crtc| gone_crtcs.contains(&crtc));
        // A plane that cannot be put back is disabled instead: FB_ID 0 and
        // CRTC_ID 0 are always valid, and the kernel ignores the geometry
        // of a plane with no framebuffer.
        push_props(
            &mut entries,
            plane,
            if clear { &["FB_ID", "CRTC_ID"] } else { &[] },
        );
    }

    entries
}

/// Add one captured object's properties to the commit, with the named ones
/// cleared to `0` instead of replayed.
fn push_props(entries: &mut Vec<RestoreEntry>, object: &CapturedObject, clear: &[&str]) {
    for prop in &object.props {
        entries.push(RestoreEntry {
            object: object.object,
            property: prop.property,
            value: if clear.contains(&prop.name.as_str()) {
                0
            } else {
                prop.value
            },
        });
    }
}

/// Create a mapping of property names and handles for given handles of a given drm resource type.
/// You may use this to easily lookup properties by name instead of going through this procedure manually.
pub(in crate::backend::drm) fn map_props<D, T>(
    fd: &D,
    handles: &[T],
    mapping: &mut HashMap<T, HashMap<String, property::Handle>>,
) -> Result<(), Error>
where
    D: DevPath + ControlDevice,
    T: ResourceHandle + Eq + std::hash::Hash,
{
    handles
        .iter()
        .map(|x| (x, fd.get_properties(*x)))
        .try_for_each(|(handle, props)| {
            let mut map = HashMap::new();
            match props {
                Ok(props) => {
                    let (prop_handles, _) = props.as_props_and_values();
                    for prop in prop_handles {
                        if let Ok(info) = fd.get_property(*prop) {
                            let name = info.name().to_string_lossy().into_owned();
                            map.insert(name, *prop);
                        }
                    }
                    mapping.insert(*handle, map);
                    Ok(())
                }
                Err(err) => Err(err),
            }
        })
        .map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error reading properties",
                dev: fd.dev_path(),
                source,
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    fn raw(id: u32) -> RawResourceHandle {
        NonZeroU32::new(id).unwrap()
    }

    fn prop(id: u32) -> property::Handle {
        property::Handle::from(raw(id))
    }

    fn crtc_handle(id: u32) -> crtc::Handle {
        crtc::Handle::from(raw(id))
    }

    fn plane_handle(id: u32) -> plane::Handle {
        plane::Handle::from(raw(id))
    }

    fn captured_object(object: RawResourceHandle, props: &[(&str, u32, u64)]) -> CapturedObject {
        CapturedObject {
            object,
            props: props
                .iter()
                .map(|&(name, property, value)| CapturedProp {
                    name: name.to_owned(),
                    property: prop(property),
                    value,
                })
                .collect(),
        }
    }

    fn entry(object: u32, property: u32, value: u64) -> RestoreEntry {
        RestoreEntry {
            object: raw(object),
            property: prop(property),
            value,
        }
    }

    /// The properties the atomic uAPI refuses or cannot replay are not
    /// captured, so the restore commit never carries them. The kernel's
    /// reasons are in the doc comment on [`capture_action`]; this test pins
    /// the decision per case.
    #[test]
    fn the_capture_takes_only_what_the_atomic_uapi_accepts_back() {
        use CaptureAction::*;

        // DPMS: drm_atomic_connector_set_property rejects it by name,
        // "legacy ... can only be set via legacy uAPI".
        assert_eq!(capture_action("DPMS", true, false, false), Skip);

        // Immutable: a connector's EDID, a plane's type, IN_FORMATS —
        // refused by drm_property_change_valid_get.
        assert_eq!(capture_action("EDID", false, false, true), Skip);
        assert_eq!(capture_action("type", false, false, false), Skip);
        assert_eq!(capture_action("IN_FORMATS", false, false, true), Skip);

        // The ATOMIC flag marks fence properties, whose values are file
        // descriptors and addresses rather than replayable state.
        assert_eq!(capture_action("IN_FENCE_FD", true, true, false), Skip);
        assert_eq!(capture_action("OUT_FENCE_PTR", true, true, false), Skip);

        // MODE_ID's blob id dies with our own first modeset; the mode is
        // captured through the legacy CRTC ioctl instead.
        assert_eq!(capture_action("MODE_ID", true, false, true), Skip);

        // Blob ids can go stale while we run; the restore commits a copy.
        assert_eq!(capture_action("GAMMA_LUT", true, false, true), CopyBlob);
        assert_eq!(capture_action("HDR_OUTPUT_METADATA", true, false, true), CopyBlob);

        // Everything else mutable is replayed as captured.
        assert_eq!(capture_action("CRTC_ID", true, false, false), Raw);
        assert_eq!(capture_action("ACTIVE", true, false, false), Raw);
        assert_eq!(capture_action("FB_ID", true, false, false), Raw);
        assert_eq!(capture_action("VRR_ENABLED", true, false, false), Raw);
    }

    /// One device: a getty on one CRTC (connector 31 bound to CRTC 42, one
    /// primary plane), plus a second CRTC that was scanning out nothing.
    fn captured_device() -> PreviousState {
        PreviousState {
            connectors: vec![captured_object(
                raw(31),
                &[
                    ("CRTC_ID", 1, 42),   // bound to CRTC 42
                    ("Colorspace", 2, 4), // mutable, replayed raw
                ],
            )],
            crtcs: vec![
                CapturedCrtc {
                    object: captured_object(
                        raw(42),
                        &[
                            ("ACTIVE", 3, 1),
                            // MODE_ID is not captured as a prop; the mode
                            // comes from the legacy CRTC snapshot below.
                        ],
                    ),
                    // A blob of our own, made at capture time.
                    mode: Some((prop(4), 99)),
                },
                CapturedCrtc {
                    object: captured_object(raw(50), &[("ACTIVE", 3, 0)]),
                    // The CRTC was scanning out nothing: MODE_ID 0.
                    mode: Some((prop(4), 0)),
                },
            ],
            planes: vec![
                captured_object(
                    raw(60),
                    &[
                        ("FB_ID", 5, 7),    // the getty's framebuffer
                        ("CRTC_ID", 1, 42), // bound to CRTC 42
                        ("SRC_W", 6, 2560 << 16),
                        ("CRTC_W", 7, 2560),
                    ],
                ),
                // An overlay plane the getty was not using.
                captured_object(
                    raw(61),
                    &[
                        ("FB_ID", 5, 0),
                        ("CRTC_ID", 1, 0),
                        ("SRC_W", 6, 0),
                        ("CRTC_W", 7, 0),
                    ],
                ),
            ],
        }
    }

    /// Everything captured is committed: the connector's binding, the
    /// CRTC's ACTIVE, the recreated MODE_ID blob, and the planes — so the
    /// commit is exactly what the kernel needs to put the getty back on.
    #[test]
    fn the_restore_replays_the_captured_state() {
        let state = captured_device();
        let entries = restore_entries(&state, &[], &[]);

        // The connector, bound back to its CRTC.
        assert!(entries.contains(&entry(31, 1, 42)));
        assert!(entries.contains(&entry(31, 2, 4)));

        // The CRTC: active, with the mode blob captured at creation
        // (not the pre-session MODE_ID id, which may be dead).
        assert!(entries.contains(&entry(42, 3, 1)));
        assert!(entries.contains(&entry(42, 4, 99)));

        // The primary plane, scanning out the getty's framebuffer again.
        assert!(entries.contains(&entry(60, 5, 7)));
        assert!(entries.contains(&entry(60, 1, 42)));
        assert!(entries.contains(&entry(60, 6, 2560 << 16)));
        assert!(entries.contains(&entry(60, 7, 2560)));

        // The CRTC that was scanning out nothing is restored to exactly
        // that: ACTIVE 0, MODE_ID 0.
        assert!(entries.contains(&entry(50, 3, 0)));
        assert!(entries.contains(&entry(50, 4, 0)));

        // No framebuffer object is ever a target of the commit: the
        // captured entries only name connectors, CRTCs and planes.
        assert!(entries.iter().all(|e| e.object != raw(7)));
    }

    /// A CRTC that cannot be restored takes itself, its connector and its
    /// planes out of the commit — the commit stays valid, and the rest of
    /// the device is still restored.
    #[test]
    fn a_crtc_that_cannot_come_back_does_not_take_the_rest_with_it() {
        let state = captured_device();
        let entries = restore_entries(&state, &[crtc_handle(42)], &[]);

        // The connector is left out entirely rather than pointed at a CRTC
        // the commit does not enable, and the CRTC itself is not half-set.
        assert!(!entries.contains(&entry(31, 1, 42)));
        assert!(!entries.contains(&entry(42, 3, 1)));
        assert!(!entries.contains(&entry(42, 4, 99)));

        // Its plane is disabled: FB_ID 0 and CRTC_ID 0 are always valid,
        // and the geometry of a plane without a framebuffer is ignored.
        assert!(entries.contains(&entry(60, 5, 0)));
        assert!(entries.contains(&entry(60, 1, 0)));
        assert!(entries.contains(&entry(60, 6, 2560 << 16)));

        // The CRTC that was off is still restored to off.
        assert!(entries.contains(&entry(50, 3, 0)));
        assert!(entries.contains(&entry(50, 4, 0)));
    }

    /// A plane whose captured framebuffer no longer exists — a previous
    /// compositor's, whose fd close destroyed it — is disabled instead of
    /// replayed, because one dead object id fails the whole commit.
    #[test]
    fn a_plane_with_a_dead_framebuffer_is_cleared_not_replayed() {
        let state = captured_device();
        let entries = restore_entries(&state, &[], &[plane_handle(60)]);

        assert!(entries.contains(&entry(60, 5, 0)));
        assert!(entries.contains(&entry(60, 1, 0)));
        // The connector and the CRTC are restored regardless.
        assert!(entries.contains(&entry(31, 1, 42)));
        assert!(entries.contains(&entry(42, 4, 99)));
    }
}
