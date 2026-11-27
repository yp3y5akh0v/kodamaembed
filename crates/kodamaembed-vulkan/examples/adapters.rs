//! Lists the Vulkan GPUs KodamaEmbed can use and what each supports.

use kodamaembed_vulkan::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for adapter in Device::adapters()? {
        let kind = if adapter.discrete {
            "discrete"
        } else {
            "integrated"
        };
        let driver = if adapter.portability {
            ", portability driver"
        } else {
            ""
        };
        match Device::new(adapter.index) {
            Ok(device) => println!(
                "{}: {} ({kind}{driver}), subgroup {}, cooperative matrices {}",
                adapter.index,
                device.name,
                device.subgroup_size,
                if device.cooperative_matrix {
                    "yes"
                } else {
                    "no"
                }
            ),
            Err(error) => println!("{}: {} unusable: {error}", adapter.index, adapter.name),
        }
    }
    Ok(())
}
