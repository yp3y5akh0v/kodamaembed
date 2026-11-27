//! KodamaEmbed's worker threads; on Windows they ask, best effort, not to be throttled.

use rayon::ThreadPoolBuildError;
#[cfg(target_os = "linux")]
use std::collections::BTreeSet;
use std::sync::LazyLock;

/// A pool of worker threads.
#[cfg(not(target_family = "wasm"))]
pub type Workers = rayon::ThreadPool;

/// Rayon's own pool: a wasm program starts no threads, and without any that pool runs every
/// task on its caller.
#[cfg(target_family = "wasm")]
pub struct Workers;

#[cfg(target_family = "wasm")]
impl Workers {
    /// Runs `op`; its parallel loops go to Rayon's pool.
    pub fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        op()
    }

    /// The threads of Rayon's pool: one where the target has none.
    pub fn current_num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

/// The CPU backend's threads, one per logical processor the process may use.
pub(crate) static POOL: LazyLock<Workers> = LazyLock::new(|| {
    worker_pool("kodamaembed-cpu", allowed_processors()).expect("worker threads start")
});

/// [`short_pool_threads`] for the allowed cores, else none.
pub(crate) static SHORT_POOL: LazyLock<Option<Workers>> = LazyLock::new(|| {
    let threads = short_pool_threads(allowed_cores()?, POOL.current_num_threads())?;
    Some(worker_pool("kodamaembed-cpu-short", threads).expect("short-input threads start"))
});

/// Physical cores the process may use, by class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cores {
    pub(crate) performance: usize,
    pub(crate) efficiency: usize,
}

/// Short inputs: performance cores alone beat them joined by up to twice as many efficiency cores.
pub(crate) fn short_pool_threads(cores: Cores, pool: usize) -> Option<usize> {
    let Cores {
        performance,
        efficiency,
    } = cores;
    (performance > 0 && efficiency <= 2 * performance && performance < pool).then_some(performance)
}

/// RelationProcessorCore in LOGICAL_PROCESSOR_RELATIONSHIP.
#[cfg(any(windows, test))]
const PROCESSOR_CORE: i32 = 0;

/// The system's fastest efficiency class marks performance cores.
#[cfg(windows)]
pub(crate) fn allowed_cores() -> Option<Cores> {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: i32,
            buffer: *mut c_void,
            length: *mut u32,
        ) -> i32;
    }
    let affinity = process_affinity()?;
    let mut length = 0_u32;
    // SAFETY: a null buffer of length 0 asks only for the length needed.
    unsafe { GetLogicalProcessorInformationEx(PROCESSOR_CORE, std::ptr::null_mut(), &mut length) };
    let mut buffer = vec![0_u64; (length as usize).div_ceil(8)];
    // SAFETY: the buffer holds `length` bytes at 8-byte alignment.
    let filled = unsafe {
        GetLogicalProcessorInformationEx(PROCESSOR_CORE, buffer.as_mut_ptr().cast(), &mut length)
    };
    if filled == 0 {
        return None;
    }
    // SAFETY: the call wrote `length` bytes into the buffer.
    let bytes =
        unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), length as usize) };
    count_core_records(bytes, |group, mask| match affinity {
        Affinity::Group(own, allowed) => group == own && mask & allowed != 0,
        Affinity::Every => true,
    })
}

/// Cores with an allowed processor; the fastest efficiency class marks performance cores.
#[cfg(any(windows, test))]
fn count_core_records(records: &[u8], allowed: impl Fn(u16, usize) -> bool) -> Option<Cores> {
    // Records: Relationship and Size (u32 each), Flags, EfficiencyClass, 20 reserved bytes and
    // GroupCount (u16); then GROUP_AFFINITY entries: Mask (pointer-sized), Group (u16), 6 reserved.
    const MASK: usize = size_of::<usize>();
    const ENTRY: usize = MASK + 8;
    let mut cores = Vec::new();
    let mut rest = records;
    while !rest.is_empty() {
        let size = u32::from_le_bytes(rest.get(4..8)?.try_into().ok()?) as usize;
        let record = rest.get(..size).filter(|_| size >= 32)?;
        rest = &rest[size..];
        if i32::from_le_bytes(record[..4].try_into().ok()?) != PROCESSOR_CORE {
            continue;
        }
        let groups = usize::from(u16::from_le_bytes(record[30..32].try_into().ok()?));
        let (entries, _) = record.get(32..32 + groups * ENTRY)?.as_chunks::<ENTRY>();
        let mut usable = false;
        for entry in entries {
            let mask = usize::from_le_bytes(entry[..MASK].try_into().ok()?);
            let group = u16::from_le_bytes(entry[MASK..MASK + 2].try_into().ok()?);
            usable |= allowed(group, mask);
        }
        cores.push((record[9], usable));
    }
    let fastest = cores.iter().map(|&(class, _)| class).max()?;
    let count = |fast: bool| {
        cores
            .iter()
            .filter(|&&(class, usable)| usable && (class == fastest) == fast)
            .count()
    };
    Some(Cores {
        performance: count(true),
        efficiency: count(false),
    })
}

#[cfg(windows)]
#[derive(Clone, Copy)]
enum Affinity {
    /// A processor group and the mask of its processors the process may use.
    Group(u16, usize),
    /// Threads in several groups, for which Windows reports no mask.
    Every,
}

#[cfg(windows)]
fn process_affinity() -> Option<Affinity> {
    use std::ffi::c_void;

    /// PROCESSOR_NUMBER.
    #[repr(C)]
    #[derive(Default)]
    struct Processor {
        group: u16,
        _number: u8,
        _reserved: u8,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessAffinityMask(
            process: *mut c_void,
            mask: *mut usize,
            system: *mut usize,
        ) -> i32;
        fn GetCurrentProcessorNumberEx(processor: *mut Processor);
    }
    let (mut mask, mut system) = (0, 0);
    // SAFETY: GetCurrentProcess returns a pseudo-handle, and both masks are writable.
    if unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut mask, &mut system) } == 0 {
        return None;
    }
    if mask == 0 {
        return Some(Affinity::Every);
    }
    // A mask means the process is in one group, so the processor this thread runs on is too.
    let mut processor = Processor::default();
    // SAFETY: `processor` is a writable PROCESSOR_NUMBER.
    unsafe { GetCurrentProcessorNumberEx(&mut processor) };
    Some(Affinity::Group(processor.group, mask))
}

/// Rust's count ignores Windows affinity masks; 0 leaves Rayon's.
#[cfg(windows)]
fn allowed_processors() -> usize {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetActiveProcessorCount(group: u16) -> u32;
    }
    const ALL_PROCESSOR_GROUPS: u16 = 0xFFFF;
    match process_affinity() {
        Some(Affinity::Group(_, mask)) => mask.count_ones() as usize,
        // SAFETY: the call reads only its argument.
        Some(Affinity::Every) => unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) as usize },
        None => 0,
    }
}

/// Rayon's count, which follows the process's affinity outside Windows.
#[cfg(not(windows))]
fn allowed_processors() -> usize {
    0
}

/// From `/sys/devices/cpu_core/cpus` on hybrid Intel processors: frequencies
/// cannot tell the classes apart, as favored cores report higher maxima.
#[cfg(target_os = "linux")]
pub(crate) fn allowed_cores() -> Option<Cores> {
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let status = read("/proc/self/status")?;
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    count_cores(
        &read("/sys/devices/system/cpu/online")?,
        allowed,
        read("/sys/devices/cpu_core/cpus").as_deref(),
        |cpu| {
            read(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list"
            ))
        },
    )
}

/// Cores with an allowed online processor; without a performance list, every core is one.
#[cfg(target_os = "linux")]
fn count_cores(
    online: &str,
    allowed: &str,
    performance: Option<&str>,
    siblings: impl Fn(usize) -> Option<String>,
) -> Option<Cores> {
    let (online, allowed) = (cpu_list(online)?, cpu_list(allowed)?);
    let performance = match performance {
        Some(listed) => Some(cpu_list(listed)?),
        None => None,
    };
    let (mut fast, mut slow) = (BTreeSet::new(), BTreeSet::new());
    for cpu in allowed.into_iter().filter(|cpu| online.contains(cpu)) {
        let core = siblings(cpu)?.trim().to_owned();
        if performance
            .as_ref()
            .is_none_or(|listed| listed.contains(&cpu))
        {
            fast.insert(core);
        } else {
            slow.insert(core);
        }
    }
    Some(Cores {
        performance: fast.len(),
        efficiency: slow.len(),
    })
}

/// The processors of a sysfs CPU list such as `0-7,16,18-19`.
#[cfg(target_os = "linux")]
fn cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',') {
        match part.split_once('-') {
            Some((first, last)) => cpus.extend(first.parse::<usize>().ok()?..=last.parse().ok()?),
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// Performance levels 0 and 1, fastest first (Intel Macs: 0 only); macOS has no affinity masks.
#[cfg(target_os = "macos")]
pub(crate) fn allowed_cores() -> Option<Cores> {
    let efficiency = match sysctl_count(c"hw.nperflevels")? {
        1 => 0,
        _ => sysctl_count(c"hw.perflevel1.physicalcpu")?,
    };
    Some(Cores {
        performance: sysctl_count(c"hw.perflevel0.physicalcpu")?,
        efficiency,
    })
}

#[cfg(target_os = "macos")]
fn sysctl_count(name: &std::ffi::CStr) -> Option<usize> {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            old_length: *mut usize,
            new: *mut c_void,
            new_length: usize,
        ) -> c_int;
    }
    let mut count: c_int = 0;
    let mut length = size_of::<c_int>();
    // SAFETY: `count` holds `length` bytes, and no new value is passed.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&raw mut count).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || length != size_of::<c_int>() {
        return None;
    }
    usize::try_from(count).ok()
}

/// No topology source elsewhere: short inputs use [`POOL`].
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub(crate) fn allowed_cores() -> Option<Cores> {
    None
}

/// `threads` workers named `name-0`, `name-1`, ... (0: one per logical processor).
#[cfg(not(target_family = "wasm"))]
pub fn worker_pool(name: &'static str, threads: usize) -> Result<Workers, ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |index| format!("{name}-{index}"))
        .start_handler(|_| raise_quality_of_service())
        .build()
}

/// Rayon's own pool whatever `threads` asks: a caller that needs a count compares it.
#[cfg(target_family = "wasm")]
pub fn worker_pool(_name: &'static str, _threads: usize) -> Result<Workers, ThreadPoolBuildError> {
    Ok(Workers)
}

#[cfg(not(target_family = "wasm"))]
fn raise_quality_of_service() {
    #[cfg(windows)]
    {
        use std::ffi::c_void;

        /// THREAD_POWER_THROTTLING_STATE.
        #[repr(C)]
        struct State {
            version: u32,
            control_mask: u32,
            state_mask: u32,
        }
        const CURRENT_VERSION: u32 = 1;
        const EXECUTION_SPEED: u32 = 0x1;
        /// ThreadPowerThrottling in THREAD_INFORMATION_CLASS.
        const POWER_THROTTLING: i32 = 3;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut c_void;
            fn SetThreadInformation(
                thread: *mut c_void,
                class: i32,
                information: *const c_void,
                size: u32,
            ) -> i32;
        }

        // The control bit set with the state bit clear asks for high quality of service.
        let state = State {
            version: CURRENT_VERSION,
            control_mask: EXECUTION_SPEED,
            state_mask: 0,
        };
        // SAFETY: `state` has the size given and outlives the call.
        let _ = unsafe {
            SetThreadInformation(
                GetCurrentThread(),
                POWER_THROTTLING,
                (&raw const state).cast(),
                size_of::<State>() as u32,
            )
        };
        // Best effort: older Windows versions and restricted hosts can refuse it.
    }
}

#[cfg(test)]
mod tests {
    use super::Cores;

    fn cores(performance: usize, efficiency: usize) -> Cores {
        Cores {
            performance,
            efficiency,
        }
    }

    /// A hybrid processor's masks, one thread per core; then one core class with and without SMT.
    #[test]
    fn short_pool_follows_the_core_ratio() {
        for (allowed, pool, short) in [
            (cores(6, 8), 14, Some(6)),
            (cores(4, 8), 12, Some(4)),
            (cores(2, 2), 4, Some(2)),
            (cores(2, 8), 10, None),
            (cores(1, 4), 5, None),
            (cores(0, 8), 8, None),
            (cores(6, 0), 6, None),
            (cores(6, 0), 12, Some(6)),
        ] {
            assert_eq!(
                super::short_pool_threads(allowed, pool),
                short,
                "{allowed:?}"
            );
        }
    }

    /// A processor-core record: its efficiency class, then one group affinity in group 0.
    fn core_record(class: u8, mask: usize) -> Vec<u8> {
        let mut record = vec![0; 40 + size_of::<usize>()];
        let size = record.len() as u32;
        record[4..8].copy_from_slice(&size.to_le_bytes());
        record[9] = class;
        record[30..32].copy_from_slice(&1_u16.to_le_bytes());
        record[32..32 + size_of::<usize>()].copy_from_slice(&mask.to_le_bytes());
        record
    }

    /// Four two-processor cores of class 1 (0-7) and four of class 0 (8-11).
    #[test]
    fn windows_core_records_count_only_allowed_processors() {
        let records: Vec<u8> = (0..4)
            .map(|core| core_record(1, 0b11 << (2 * core)))
            .chain((8..12).map(|processor| core_record(0, 1 << processor)))
            .flatten()
            .collect();
        let count = |allowed: usize| {
            super::count_core_records(&records, |group, mask| group == 0 && mask & allowed != 0)
        };
        assert_eq!(count(0xFFF), Some(cores(4, 4)));
        assert_eq!(count(0b1000_0101), Some(cores(3, 0)));
        assert_eq!(count(0b11_0000_0010), Some(cores(1, 2)));
        assert_eq!(count(0xF00), Some(cores(0, 4)));
        let alike: Vec<u8> = (0..4)
            .flat_map(|core| core_record(0, 0b11 << (2 * core)))
            .collect();
        assert_eq!(
            super::count_core_records(&alike, |_, _| true),
            Some(cores(4, 0))
        );
    }

    /// An i5-13500: six two-thread performance cores (0-11), eight efficiency cores.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_cores_count_only_allowed_processors() {
        let siblings = |cpu: usize| {
            Some(if cpu < 12 {
                format!("{}-{}\n", cpu & !1, cpu | 1)
            } else {
                format!("{cpu}\n")
            })
        };
        let listed = Some("0-11\n");
        for (online, allowed, performance, counted) in [
            ("0-19\n", "0-19\n", listed, cores(6, 8)),
            ("0-19\n", "0,2,4,7,12-19\n", listed, cores(4, 8)),
            ("0-19\n", "1,12-15\n", listed, cores(1, 4)),
            ("0-19\n", "12-17\n", listed, cores(0, 6)),
            ("0-15\n", "0-19\n", listed, cores(6, 4)),
            ("0-19\n", "0-19\n", None, cores(14, 0)),
        ] {
            let count = super::count_cores(online, allowed, performance, siblings);
            assert_eq!(count, Some(counted), "{online} {allowed}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macs_report_their_cores() {
        let logical = std::thread::available_parallelism().unwrap().get();
        let cores = super::allowed_cores().unwrap();
        assert!(cores.performance > 0 && cores.performance + cores.efficiency <= logical);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_lists_expand_ranges_and_single_processors() {
        assert_eq!(
            super::cpu_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(super::cpu_list("5"), Some(vec![5]));
        assert_eq!(super::cpu_list(""), None);
        assert_eq!(super::cpu_list("0-x"), None);
    }
}
