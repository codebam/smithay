use std::os::unix::io::OwnedFd;
use std::{
    fmt,
    sync::{Arc, Mutex},
};

use tracing::debug;
use wayland_protocols_misc::zwp_virtual_keyboard_v1::server::zwp_virtual_keyboard_v1::Error::NoKeymap;
use wayland_protocols_misc::zwp_virtual_keyboard_v1::server::zwp_virtual_keyboard_v1::{
    self, ZwpVirtualKeyboardV1,
};
use wayland_server::{
    Client, DataInit, DisplayHandle, Resource,
    protocol::wl_keyboard::{KeyState, KeymapFormat},
};
use xkbcommon::xkb;

use crate::input::keyboard::{KeyboardTarget, KeymapFile, ModifiersState};
use super::VirtualKeyboardKeyFilter;
use crate::{
    input::{Seat, SeatHandler},
    utils::SERIAL_COUNTER,
    wayland::{
        Dispatch2,
        seat::{WaylandFocus, keyboard::for_each_focused_kbds},
    },
};

#[derive(Debug, Default)]
pub(crate) struct VirtualKeyboard {
    state: Option<VirtualKeyboardState>,
}

struct VirtualKeyboardState {
    keymap: KeymapFile,
    mods: ModifiersState,
    state: xkb::State,
}

impl fmt::Debug for VirtualKeyboardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtualKeyboardState")
            .field("keymap", &self.keymap)
            .field("mods", &self.mods)
            .field("state", &self.state.get_raw_ptr())
            .finish()
    }
}

// This is OK because all parts of `xkb` will remain on the
// same thread
unsafe impl Send for VirtualKeyboard {}

/// Handle to a virtual keyboard instance
#[derive(Debug, Clone, Default)]
pub(crate) struct VirtualKeyboardHandle {
    pub(crate) inner: Arc<Mutex<VirtualKeyboard>>,
}

/// User data of ZwpVirtualKeyboardV1 object
pub struct VirtualKeyboardUserData<D: SeatHandler> {
    pub(super) handle: VirtualKeyboardHandle,
    pub(crate) seat: Seat<D>,
}

impl<D: SeatHandler> fmt::Debug for VirtualKeyboardUserData<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtualKeyboardUserData")
            .field("handle", &self.handle)
            .field("seat", &self.seat.arc)
            .finish()
    }
}

impl<D> Dispatch2<ZwpVirtualKeyboardV1, D> for VirtualKeyboardUserData<D>
where
    D: SeatHandler + VirtualKeyboardKeyFilter + 'static,
    <D as SeatHandler>::KeyboardFocus: WaylandFocus,
{
    fn request(
        &self,
        user_data: &mut D,
        _client: &Client,
        virtual_keyboard: &ZwpVirtualKeyboardV1,
        request: zwp_virtual_keyboard_v1::Request,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            zwp_virtual_keyboard_v1::Request::Keymap { format, fd, size } => {
                update_keymap(self, format, fd, size as usize);
            }
            zwp_virtual_keyboard_v1::Request::Key { time, key, state } => {
                // This should be wl_keyboard::KeyState, but the protocol does not state
                // the parameter is an enum.
                let key_state = if state == 1 {
                    KeyState::Pressed
                } else {
                    KeyState::Released
                };

                // The compositor first, before anything is sent anywhere.
                //
                // The keysym has to be resolved here, against this keyboard's
                // own keymap: the client chose the keycode out of the keymap it
                // uploaded, and the seat's keymap is a different one that maps
                // the same code to something else. Resolving it later, or
                // elsewhere, gets the wrong key.
                //
                // The lock is dropped first. Whatever the compositor does with
                // the key is likely to reach back into this seat, and holding
                // this keyboard's lock across that is a deadlock waiting for a
                // binding that touches it.
                let intercepted = {
                    let mut virtual_data = self.handle.inner.lock().unwrap();
                    let Some(vk_state) = virtual_data.state.as_mut() else {
                        virtual_keyboard.post_error(NoKeymap, "`key` sent before keymap.");
                        return;
                    };
                    // Evdev keycodes are offset by 8 in XKB, as everywhere else
                    // this protocol meets it.
                    let keycode = xkb::Keycode::new(key + 8);
                    let keysym = vk_state.state.key_get_one_sym(keycode);
                    // The symbol on the key rather than the one the modifiers
                    // make of it, for matching the key part of a chord. Level 0
                    // of the effective layout — a virtual keyboard uploads its
                    // own keymap, so there is no other layout to fall back to
                    // the way `KeysymHandle` does for a Cyrillic or Greek one.
                    let layout = vk_state.state.key_get_layout(keycode);
                    let raw_keysym = vk_state
                        .state
                        .get_keymap()
                        .key_get_syms_by_level(keycode, layout, 0)
                        .first()
                        .copied();
                    let mods = vk_state.mods;
                    drop(virtual_data);
                    user_data.virtual_keyboard_key(
                        &self.seat,
                        keysym,
                        raw_keysym,
                        mods,
                        key,
                        key_state,
                        time,
                    )
                };
                if intercepted {
                    return;
                }

                // Ensure keymap was initialized.
                let mut virtual_data = self.handle.inner.lock().unwrap();
                let vk_state = match virtual_data.state.as_mut() {
                    Some(vk_state) => vk_state,
                    None => {
                        virtual_keyboard.post_error(NoKeymap, "`key` sent before keymap.");
                        return;
                    }
                };

                // Ensure virtual keyboard's keymap is active.
                let keyboard_handle = self.seat.get_keyboard().unwrap();
                let mut internal = keyboard_handle.arc.internal.lock().unwrap();
                let focus = internal.focus.as_mut().map(|(focus, _)| focus);
                keyboard_handle.send_keymap(user_data, &focus, &vk_state.keymap, vk_state.mods);

                if let Some(wl_surface) = focus.and_then(|f| f.wl_surface()) {
                    for_each_focused_kbds(&self.seat, &wl_surface, |kbd| {
                        kbd.key(SERIAL_COUNTER.next_serial().0, time, key, key_state);
                    });
                }
            }
            zwp_virtual_keyboard_v1::Request::Modifiers {
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
            } => {
                // Ensure keymap was initialized.
                let mut virtual_data = self.handle.inner.lock().unwrap();
                let state = match virtual_data.state.as_mut() {
                    Some(state) => state,
                    None => {
                        virtual_keyboard.post_error(NoKeymap, "`modifiers` sent before keymap.");
                        return;
                    }
                };

                // Update virtual keyboard's modifier state.
                state
                    .state
                    .update_mask(mods_depressed, mods_latched, mods_locked, 0, 0, group);
                state.mods.update_with(&state.state);

                // Ensure virtual keyboard's keymap is active.
                let keyboard_handle = self.seat.get_keyboard().unwrap();
                let mut internal = keyboard_handle.arc.internal.lock().unwrap();
                let focus = internal.focus.as_mut().map(|(focus, _)| focus);
                let keymap_changed =
                    keyboard_handle.send_keymap(user_data, &focus, &state.keymap, state.mods);

                // Report modifiers change to all keyboards.
                if !keymap_changed {
                    if let Some(focus) = focus {
                        focus.modifiers(&self.seat, user_data, state.mods, SERIAL_COUNTER.next_serial());
                    }
                }
            }
            zwp_virtual_keyboard_v1::Request::Destroy => {
                // Nothing to do
            }
            _ => unreachable!(),
        }
    }
}

/// Handle the zwp_virtual_keyboard_v1::keymap request.
///
/// The `true` returns when keymap was properly loaded.
fn update_keymap<D>(data: &VirtualKeyboardUserData<D>, format: u32, fd: OwnedFd, size: usize)
where
    D: SeatHandler + 'static,
{
    // Only libxkbcommon compatible keymaps are supported.
    if format != KeymapFormat::XkbV1 as u32 {
        debug!("Unsupported keymap format: {format:?}");
        return;
    }

    let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
    // SAFETY: we can map the keymap into the memory.
    let new_keymap = match unsafe {
        xkb::Keymap::new_from_fd(
            &context,
            fd,
            size,
            xkb::KEYMAP_FORMAT_TEXT_V1,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
    } {
        Ok(Some(new_keymap)) => new_keymap,
        Ok(None) => {
            debug!("Invalid libxkbcommon keymap");
            return;
        }
        Err(err) => {
            debug!("Could not map the keymap: {err:?}");
            return;
        }
    };

    // Store active virtual keyboard map.
    let mut inner = data.handle.inner.lock().unwrap();
    let mods = inner.state.take().map(|state| state.mods).unwrap_or_default();
    inner.state = Some(VirtualKeyboardState {
        mods,
        keymap: KeymapFile::new(&new_keymap),
        state: xkb::State::new(&new_keymap),
    });
}
