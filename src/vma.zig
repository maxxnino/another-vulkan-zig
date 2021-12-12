const vk = @import("vulkan");

pub fn createAllocator(info: *const VmaAllocatorCreateInfo) !VmaAllocator {
    var allocator: VmaAllocator = undefined;
    const result = vmaCreateAllocator(info, &allocator);
    switch (result) {
        vk.Result.success => {},
        vk.Result.error_out_of_host_memory => return error.OutOfHostMemory,
        vk.Result.error_out_of_device_memory => return error.OutOfDeviceMemory,
        vk.Result.error_initialization_failed => return error.InitializationFailed,
        vk.Result.error_layer_not_present => return error.LayerNotPresent,
        vk.Result.error_extension_not_present => return error.ExtensionNotPresent,
        vk.Result.error_incompatible_driver => return error.IncompatibleDriver,
        else => return error.Unknown,
    }
    return allocator;
}
pub extern fn vmaCreateAllocator(pCreateInfo: *const VmaAllocatorCreateInfo, pAllocator: *VmaAllocator) vk.Result;

pub const VmaAllocator = enum(usize) { null_handle = 0, _ };
pub const VmaVulkanFunctions = extern struct {
    vkGetInstanceProcAddr: vk.PfnGetInstanceProcAddr,
    vkGetDeviceProcAddr: vk.PfnGetDeviceProcAddr,
    vkGetPhysicalDeviceProperties: vk.PfnGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceMemoryProperties: vk.PfnGetPhysicalDeviceMemoryProperties,
    vkAllocateMemory: vk.PfnAllocateMemory,
    vkFreeMemory: vk.PfnFreeMemory,
    vkMapMemory: vk.PfnMapMemory,
    vkUnmapMemory: vk.PfnUnmapMemory,
    vkFlushMappedMemoryRanges: vk.PfnFlushMappedMemoryRanges,
    vkInvalidateMappedMemoryRanges: vk.PfnInvalidateMappedMemoryRanges,
    vkBindBufferMemory: vk.PfnBindBufferMemory,
    vkBindImageMemory: vk.PfnBindImageMemory,
    vkGetBufferMemoryRequirements: vk.PfnGetBufferMemoryRequirements,
    vkGetImageMemoryRequirements: vk.PfnGetImageMemoryRequirements,
    vkCreateBuffer: vk.PfnCreateBuffer,
    vkDestroyBuffer: vk.PfnDestroyBuffer,
    vkCreateImage: vk.PfnCreateImage,
    vkDestroyImage: vk.PfnDestroyImage,
    vkCmdCopyBuffer: vk.PfnCmdCopyBuffer,
    vkGetBufferMemoryRequirements2KHR: vk.PfnGetBufferMemoryRequirements2,
    vkGetImageMemoryRequirements2KHR: vk.PfnGetImageMemoryRequirements2,
    vkBindBufferMemory2KHR: vk.PfnBindBufferMemory2,
    vkBindImageMemory2KHR: vk.PfnBindImageMemory2,
    vkGetPhysicalDeviceMemoryProperties2KHR: vk.PfnGetPhysicalDeviceProperties2,
};

pub const VmaAllocatorCreateFlags = packed struct {
    externally_synchronized_bit: bool = false,
    khr_dedicated_allocation_bit: bool = false,
    khr_bind_memory2_bit: bool = false,
    ext_memory_budget_bit: bool = false,
    amd_device_coherent_memory_bit: bool = false,
    buffer_device_address_bit: bool = false,
    ext_memory_priority_bit: bool = false,
    _reserved_bit_7: bool = false,
    _reserved_bit_8: bool = false,
    _reserved_bit_9: bool = false,
    _reserved_bit_10: bool = false,
    _reserved_bit_11: bool = false,
    _reserved_bit_12: bool = false,
    _reserved_bit_13: bool = false,
    _reserved_bit_14: bool = false,
    _reserved_bit_15: bool = false,
    _reserved_bit_16: bool = false,
    _reserved_bit_17: bool = false,
    _reserved_bit_18: bool = false,
    _reserved_bit_19: bool = false,
    _reserved_bit_20: bool = false,
    _reserved_bit_21: bool = false,
    _reserved_bit_22: bool = false,
    _reserved_bit_23: bool = false,
    _reserved_bit_24: bool = false,
    _reserved_bit_25: bool = false,
    _reserved_bit_26: bool = false,
    _reserved_bit_27: bool = false,
    _reserved_bit_28: bool = false,
    _reserved_bit_29: bool = false,
    _reserved_bit_30: bool = false,
    _reserved_bit_31: bool = false,
    pub usingnamespace vk.FlagsMixin(vk.QueueFlags, vk.Flags);
};
pub const VmaAllocatorCreateInfo = extern struct {
    flags: VmaAllocatorCreateFlags,
    physicalDevice: vk.PhysicalDevice,
    device: vk.Device,
    preferredLargeHeapBlockSize: vk.DeviceSize = 0,
    pAllocationCallbacks: ?*const vk.AllocationCallbacks = null,
    pDeviceMemoryCallbacks: ?*const VmaDeviceMemoryCallbacks = null,
    frameInUseCount: u32,
    pHeapSizeLimit: ?[*]const vk.DeviceSize = null,
    pVulkanFunctions: ?*const VmaVulkanFunctions = null,
    pRecordSettings: ?*const VmaRecordSettings = null,
    instance: vk.Instance,
    vulkanApiVersion: u32,
    pTypeExternalMemoryHandleTypes: ?[*]const vk.ExternalMemoryHandleTypeFlagsKHR = null,
};

// Garbage
pub const PFN_vmaAllocateDeviceMemoryFunction = ?fn (VmaAllocator, u32, vk.DeviceMemory, vk.DeviceSize, ?*c_void) callconv(.C) void;
pub const PFN_vmaFreeDeviceMemoryFunction = ?fn (VmaAllocator, u32, vk.DeviceMemory, vk.DeviceSize, ?*c_void) callconv(.C) void;
pub const VmaDeviceMemoryCallbacks = extern struct {
    pfnAllocate: PFN_vmaAllocateDeviceMemoryFunction,
    pfnFree: PFN_vmaFreeDeviceMemoryFunction,
    pUserData: ?*c_void,
};
pub const VMA_RECORD_FLUSH_AFTER_CALL_BIT: c_int = 1;
pub const VMA_RECORD_FLAG_BITS_MAX_ENUM: c_int = 2147483647;
pub const enum_VmaRecordFlagBits = c_uint;
pub const VmaRecordFlagBits = enum_VmaRecordFlagBits;
pub const VmaRecordFlags = VkFlags;
pub const struct_VmaRecordSettings = extern struct {
    flags: VmaRecordFlags,
    pFilePath: [*c]const u8,
};
pub const VmaRecordSettings = struct_VmaRecordSettings;
pub const VkFlags = u32;
