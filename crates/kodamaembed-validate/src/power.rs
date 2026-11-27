//! Opts the benchmark process out of Windows execution-speed throttling.

/// A no-op off Windows.
pub fn run_unthrottled() -> crate::Result<()> {
    #[cfg(windows)]
    {
        use std::ffi::c_void;

        /// PROCESS_POWER_THROTTLING_STATE.
        #[repr(C)]
        struct State {
            version: u32,
            control_mask: u32,
            state_mask: u32,
        }
        const CURRENT_VERSION: u32 = 1;
        const EXECUTION_SPEED: u32 = 0x1;
        /// ProcessPowerThrottling in PROCESS_INFORMATION_CLASS.
        const POWER_THROTTLING: i32 = 4;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut c_void;
            fn SetProcessInformation(
                process: *mut c_void,
                class: i32,
                information: *const c_void,
                size: u32,
            ) -> i32;
        }

        // The control bit set with the state bit clear turns throttling off.
        let state = State {
            version: CURRENT_VERSION,
            control_mask: EXECUTION_SPEED,
            state_mask: 0,
        };
        // SAFETY: `state` has the size given and outlives the call.
        let done = unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                POWER_THROTTLING,
                (&raw const state).cast(),
                size_of::<State>() as u32,
            )
        };
        if done == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
