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
pub extern fn vmaDestroyAllocator(allocator: VmaAllocator) void;
pub extern fn vmaCreateImage(
    allocator: VmaAllocator,
    pImageCreateInfo: *const vk.ImageCreateInfo,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    pImage: *vk.Image,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) vk.Result;
pub extern fn vmaDestroyImage(allocator: VmaAllocator, image: vk.Image, allocation: VmaAllocation) void;

pub extern fn vmaCreateBuffer(
    allocator: VmaAllocator,
    pBufferCreateInfo: *const vk.BufferCreateInfo,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    pBuffer: *vk.Buffer,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) vk.Result;
pub extern fn vmaDestroyBuffer(allocator: VmaAllocator, buffer: vk.Buffer, allocation: VmaAllocation) void;
pub extern fn vmaCreatePool(allocator: VmaAllocator, pCreateInfo: *const VmaPoolCreateInfo, pPool: *VmaPool) vk.Result;
pub extern fn vmaDestroyPool(allocator: VmaAllocator, pool: VmaPool) void;
pub extern fn vmaMakePoolAllocationsLost(allocator: VmaAllocator, pool: VmaPool, pLostAllocationCount: ?*usize) void;
pub extern fn vmaGetPoolName(allocator: VmaAllocator, pool: VmaPool, ppName: ?*[*:0]const u8) void;
pub extern fn vmaSetPoolName(allocator: VmaAllocator, pool: VmaPool, pName: ?[*:0]const u8) void;
pub extern fn vmaCheckPoolCorruption(allocator: VmaAllocator, pool: VmaPool) vk.Result;
pub extern fn vmaMapMemory(allocator: VmaAllocator, allocation: VmaAllocation, ppData: *?*c_void) vk.Result;
pub extern fn vmaUnmapMemory(allocator: VmaAllocator, allocation: VmaAllocation) void;
pub extern fn vmaSetCurrentFrameIndex(allocator: VmaAllocator, frameIndex: u32) void;
pub extern fn vmaBindBufferMemory(allocator: VmaAllocator, allocation: VmaAllocation, buffer: vk.Buffer) vk.Result;
pub extern fn vmaBindImageMemory(allocator: VmaAllocator, allocation: VmaAllocation, image: vk.Image) vk.Result;
pub extern fn vmaCheckCorruption(allocator: VmaAllocator, memoryTypeBits: u32) vk.Result;
pub extern fn vmaFlushAllocation(allocator: VmaAllocator, allocation: VmaAllocation, offset: vk.DeviceSize, size: vk.DeviceSize) vk.Result;
pub extern fn vmaInvalidateAllocation(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
) vk.Result;
pub extern fn vmaFlushAllocations(
    allocator: VmaAllocator,
    allocationCount: u32,
    allocations: [*]const VmaAllocation,
    offsets: ?[*]const vk.DeviceSize,
    sizes: ?[*]const vk.DeviceSize,
) vk.Result;
pub extern fn vmaInvalidateAllocations(
    allocator: VmaAllocator,
    allocationCount: u32,
    allocations: [*]const VmaAllocation,
    offsets: ?[*]const vk.DeviceSize,
    sizes: ?[*]const vk.DeviceSize,
) vk.Result;
pub extern fn vmaFindMemoryTypeIndex(
    allocator: VmaAllocator,
    memoryTypeBits: u32,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    pMemoryTypeIndex: *u32,
) vk.Result;
pub extern fn vmaFindMemoryTypeIndexForBufferInfo(
    allocator: VmaAllocator,
    pBufferCreateInfo: *const vk.BufferCreateInfo,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    pMemoryTypeIndex: *u32,
) vk.Result;
pub extern fn vmaFindMemoryTypeIndexForImageInfo(
    allocator: VmaAllocator,
    pImageCreateInfo: *const vk.ImageCreateInfo,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    pMemoryTypeIndex: *u32,
) vk.Result;
pub extern fn vmaBuildStatsString(allocator: VmaAllocator, ppStatsString: *[*:0]u8, detailedMap: vk.Bool32) void;
pub extern fn vmaFreeStatsString(allocator: VmaAllocator, pStatsString: [*:0]u8) void;
pub extern fn vmaGetAllocatorInfo(allocator: VmaAllocator, pAllocatorInfo: *VmaAllocatorInfo) void;
pub extern fn vmaGetPhysicalDeviceProperties(
    allocator: VmaAllocator,
    ppPhysicalDeviceProperties: **const vk.PhysicalDeviceProperties,
) void;
pub extern fn vmaGetMemoryProperties(
    allocator: VmaAllocator,
    ppPhysicalDeviceMemoryProperties: **const vk.PhysicalDeviceMemoryProperties,
) void;
pub extern fn vmaGetMemoryTypeProperties(allocator: VmaAllocator, memoryTypeIndex: u32, pFlags: *vk.MemoryPropertyFlags) void;
pub extern fn vmaBindBufferMemory2(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    allocationLocalOffset: vk.DeviceSize,
    buffer: vk.Buffer,
    pNext: ?*const c_void,
) vk.Result;
pub extern fn vmaBindImageMemory2(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    allocationLocalOffset: vk.DeviceSize,
    image: vk.Image,
    pNext: ?*const c_void,
) vk.Result;
pub extern fn vmaCreateBufferWithAlignment(
    allocator: VmaAllocator,
    pBufferCreateInfo: *const vk.BufferCreateInfo,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    minAlignment: vk.DeviceSize,
    pBuffer: *vk.Buffer,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) vk.Result;
pub extern fn vmaAllocateMemory(
    allocator: VmaAllocator,
    pVkMemoryRequirements: *const vk.MemoryRequirements,
    pCreateInfo: *const VmaAllocationCreateInfo,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) vk.Result;
pub extern fn vmaAllocateMemoryPages(
    allocator: VmaAllocator,
    pVkMemoryRequirements: [*c]const vk.MemoryRequirements,
    pCreateInfo: *const VmaAllocationCreateInfo,
    allocationCount: usize,
    pAllocations: [*]VmaAllocation,
    pAllocationInfo: ?[*]VmaAllocationInfo,
) vk.Result;
pub extern fn vmaAllocateMemoryForBuffer(
    allocator: VmaAllocator,
    buffer: vk.Buffer,
    pCreateInfo: *const VmaAllocationCreateInfo,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) vk.Result;
pub extern fn vmaAllocateMemoryForImage(
    allocator: VmaAllocator,
    image: vk.Image,
    pCreateInfo: *const VmaAllocationCreateInfo,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) vk.Result;
pub extern fn vmaFreeMemory(allocator: VmaAllocator, allocation: VmaAllocation) void;
pub extern fn vmaFreeMemoryPages(allocator: VmaAllocator, allocationCount: usize, pAllocations: [*]const VmaAllocation) void;
pub extern fn vmaGetAllocationInfo(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    pAllocationInfo: *VmaAllocationInfo,
) void;
pub extern fn vmaTouchAllocation(allocator: VmaAllocator, allocation: VmaAllocation) vk.Bool32;
pub extern fn vmaSetAllocationUserData(allocator: VmaAllocator, allocation: VmaAllocation, pUserData: ?*c_void) void;
pub extern fn vmaCreateLostAllocation(allocator: VmaAllocator, pAllocation: *VmaAllocation) void;
pub extern fn vmaGetAllocationMemoryProperties(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    pFlags: *vk.MemoryPropertyFlags,
) void;
pub extern fn vmaCalculateStats(allocator: VmaAllocator, pStats: *VmaStats) void;
pub extern fn vmaGetHeapBudgets(allocator: VmaAllocator, pBudgets: [*]VmaBudget) void;
pub extern fn vmaClearVirtualBlock(virtualBlock: VmaVirtualBlock) void;
pub extern fn vmaVirtualFree(virtualBlock: VmaVirtualBlock, offset: vk.DeviceSize) void;
pub extern fn vmaSetVirtualAllocationUserData(virtualBlock: VmaVirtualBlock, offset: vk.DeviceSize, pUserData: ?*c_void) void;
pub extern fn vmaCalculateVirtualBlockStats(virtualBlock: VmaVirtualBlock, pStatInfo: *VmaStatInfo) void;
pub extern fn vmaBuildVirtualBlockStatsString(
    virtualBlock: VmaVirtualBlock,
    ppStatsString: *[*:0]u8,
    detailedMap: vk.Bool32,
) void;
pub extern fn vmaFreeVirtualBlockStatsString(virtualBlock: VmaVirtualBlock, pStatsString: [*:0]u8) void;
pub extern fn vmaDestroyVirtualBlock(virtualBlock: VmaVirtualBlock) void;
pub extern fn vmaIsVirtualBlockEmpty(virtualBlock: VmaVirtualBlock) vk.Bool32;
pub extern fn vmaGetVirtualAllocationInfo(
    virtualBlock: VmaVirtualBlock,
    offset: vk.DeviceSize,
    pVirtualAllocInfo: *VmaVirtualAllocationInfo,
) void;
pub extern fn vmaGetPoolStats(allocator: VmaAllocator, pool: VmaPool, pPoolStats: *VmaPoolStats) void;
pub extern fn vmaVirtualAllocate(
    virtualBlock: VmaVirtualBlock,
    pCreateInfo: *const VmaVirtualAllocationCreateInfo,
    pOffset: *vk.DeviceSize,
) vk.Result;
pub extern fn vmaDefragment(
    allocator: VmaAllocator,
    pAllocations: [*c]const VmaAllocation,
    allocationCount: usize,
    pAllocationsChanged: [*]vk.Bool32,
    pDefragmentationInfo: ?*const VmaDefragmentationInfo,
    pDefragmentationStats: ?*VmaDefragmentationStats,
) vk.Result;
pub extern fn vmaDefragmentationEnd(allocator: VmaAllocator, context: VmaDefragmentationContext) vk.Result;
pub extern fn vmaEndDefragmentationPass(allocator: VmaAllocator, context: VmaDefragmentationContext) vk.Result;
pub extern fn vmaDefragmentationBegin(
    allocator: VmaAllocator,
    pInfo: *const VmaDefragmentationInfo2,
    pStats: ?*VmaDefragmentationStats,
    pContext: *VmaDefragmentationContext,
) vk.Result;
pub extern fn vmaBeginDefragmentationPass(
    allocator: VmaAllocator,
    context: VmaDefragmentationContext,
    pInfo: *VmaDefragmentationPassInfo,
) vk.Result;
pub extern fn vmaCreateVirtualBlock(
    pCreateInfo: *const VmaVirtualBlockCreateInfo,
    pVirtualBlock: *VmaVirtualBlock,
) vk.Result;

pub const VmaAllocator = enum(usize) { null_handle = 0, _ };
pub const VmaAllocation = enum(usize) { null_handle = 0, _ };
pub const VmaPool = enum(usize) { null_handle = 0, _ };
pub const VmaVirtualBlock = enum(usize) { null_handle = 0, _ };
pub const VmaDefragmentationContext = enum(usize) { null_handle = 0, _ };

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
    vkGetBufferMemoryRequirements2: vk.PfnGetBufferMemoryRequirements2,
    vkGetImageMemoryRequirements2: vk.PfnGetImageMemoryRequirements2,
    vkBindBufferMemory2KHR: vk.PfnBindBufferMemory2,
    vkBindImageMemory2KHR: vk.PfnBindImageMemory2,
    vkGetPhysicalDeviceMemoryProperties2: vk.PfnGetPhysicalDeviceProperties2,
};

pub const VmaAllocatorCreateFlags = packed struct {
    externally_synchronized_bit: bool align(@alignOf(vk.Flags)) = false,
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
    pub usingnamespace vk.FlagsMixin(VmaAllocatorCreateFlags, vk.Flags);
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

pub const VmaAllocationCreateInfo = extern struct {
    flags: VmaAllocationCreateFlags,
    usage: VmaMemoryUsage,
    requiredFlags: vk.MemoryPropertyFlags,
    preferredFlags: vk.MemoryPropertyFlags,
    memoryTypeBits: u32,
    pool: VmaPool,
    pUserData: ?*c_void,
    priority: f32,
};

pub const VmaAllocationInfo = extern struct {
    memoryType: u32,
    deviceMemory: vk.DeviceMemory,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
    pMappedData: ?*c_void,
    pUserData: ?*c_void,
};
pub const VmaPoolCreateInfo = extern struct {
    memoryTypeIndex: u32,
    flags: VmaPoolCreateFlags,
    blockSize: vk.DeviceSize,
    minBlockCount: usize,
    maxBlockCount: usize,
    frameInUseCount: u32,
    priority: f32,
    minAllocationAlignment: vk.DeviceSize,
    pMemoryAllocateNext: ?*c_void,
};
pub const VmaBudget = extern struct {
    blockBytes: vk.DeviceSize,
    allocationBytes: vk.DeviceSize,
    usage: vk.DeviceSize,
    budget: vk.DeviceSize,
};
pub const VmaStatInfo = extern struct {
    blockCount: u32,
    allocationCount: u32,
    unusedRangeCount: u32,
    usedBytes: vk.DeviceSize,
    unusedBytes: vk.DeviceSize,
    allocationSizeMin: vk.DeviceSize,
    allocationSizeAvg: vk.DeviceSize,
    allocationSizeMax: vk.DeviceSize,
    unusedRangeSizeMin: vk.DeviceSize,
    unusedRangeSizeAvg: vk.DeviceSize,
    unusedRangeSizeMax: vk.DeviceSize,
};
pub const VmaStats = extern struct {
    memoryType: [32]VmaStatInfo,
    memoryHeap: [16]VmaStatInfo,
    total: VmaStatInfo,
};
pub const VmaVirtualAllocationInfo = extern struct {
    size: vk.DeviceSize,
    pUserData: ?*c_void,
};
pub const VmaPoolStats = extern struct {
    size: vk.DeviceSize,
    unusedSize: vk.DeviceSize,
    allocationCount: usize,
    unusedRangeCount: usize,
    unusedRangeSizeMax: vk.DeviceSize,
    blockCount: usize,
};
pub const VmaVirtualAllocationCreateInfo = extern struct {
    size: vk.DeviceSize,
    alignment: vk.DeviceSize,
    flags: VmaVirtualAllocationCreateFlags,
    pUserData: ?*c_void,
};
pub const VmaDefragmentationInfo = extern struct {
    maxBytesToMove: vk.DeviceSize,
    maxAllocationsToMove: u32,
};
pub const VmaDefragmentationStats = extern struct {
    bytesMoved: vk.DeviceSize,
    bytesFreed: vk.DeviceSize,
    allocationsMoved: u32,
    deviceMemoryBlocksFreed: u32,
};
pub const VmaDefragmentationInfo2 = extern struct {
    flags: VmaDefragmentationFlags,
    allocationCount: u32,
    pAllocations: [*]const VmaAllocation,
    pAllocationsChanged: ?[*]vk.Bool32,
    poolCount: u32,
    pPools: ?[*]const VmaPool,
    maxCpuBytesToMove: vk.DeviceSize,
    maxCpuAllocationsToMove: u32,
    maxGpuBytesToMove: vk.DeviceSize,
    maxGpuAllocationsToMove: u32,
    commandBuffer: vk.CommandBuffer,
};
pub const VmaDefragmentationPassInfo = extern struct {
    moveCount: u32,
    pMoves: *VmaDefragmentationPassMoveInfo,
};
pub const VmaDefragmentationPassMoveInfo = extern struct {
    allocation: VmaAllocation,
    memory: vk.DeviceMemory,
    offset: vk.DeviceSize,
};
pub const VmaVirtualBlockCreateInfo = extern struct {
    size: vk.DeviceSize,
    flags: VmaVirtualBlockCreateFlagBits,
    pAllocationCallbacks: *const vk.AllocationCallbacks,
};
pub const VmaAllocatorInfo = extern struct {
    instance: vk.Instance,
    physicalDevice: vk.PhysicalDevice,
    device: vk.Device,
};
pub const VmaRecordSettings = extern struct {
    flags: VmaRecordFlags,
    pFilePath: [*:0]const u8,
};
pub const vmaAllocateDeviceMemoryFunction = fn (VmaAllocator, u32, vk.DeviceMemory, vk.DeviceSize, ?*c_void) callconv(.C) void;
pub const vmaFreeDeviceMemoryFunction = fn (VmaAllocator, u32, vk.DeviceMemory, vk.DeviceSize, ?*c_void) callconv(.C) void;
pub const VmaDeviceMemoryCallbacks = extern struct {
    pfnAllocate: ?vmaAllocateDeviceMemoryFunction,
    pfnFree: ?vmaFreeDeviceMemoryFunction,
    pUserData: ?*c_void,
};

// Garbage
pub const VkFlags = u32;

pub const VMA_RECORD_FLUSH_AFTER_CALL_BIT: c_int = 1;
pub const VmaRecordFlags = VkFlags;

pub const VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT: c_int = 1;
pub const VMA_VIRTUAL_BLOCK_CREATE_BUDDY_ALGORITHM_BIT: c_int = 2;
pub const VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK: c_int = 3;
pub const VmaVirtualBlockCreateFlagBits = c_uint;
pub const VmaVirtualBlockCreateFlags = VkFlags;

pub const VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT: c_int = 64;
pub const VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT: c_int = 65536;
pub const VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT: c_int = 262144;
pub const VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT: c_int = 131072;
pub const VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK: c_int = 458752;
pub const VmaVirtualAllocationCreateFlags = VkFlags;

pub const VMA_DEFRAGMENTATION_FLAG_INCREMENTAL: c_int = 1;
pub const VmaDefragmentationFlags = VkFlags;

pub const VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT: c_int = 2;
pub const VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT: c_int = 4;
pub const VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT: c_int = 8;
pub const VMA_POOL_CREATE_ALGORITHM_MASK: c_int = 12;
pub const VmaPoolCreateFlagBits = c_uint;
pub const VmaPoolCreateFlags = VkFlags;

pub const VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT: c_int = 1;
pub const VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT: c_int = 2;
pub const VMA_ALLOCATION_CREATE_MAPPED_BIT: c_int = 4;
pub const VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT: c_int = 8;
pub const VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT: c_int = 16;
pub const VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT: c_int = 32;
pub const VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT: c_int = 64;
pub const VMA_ALLOCATION_CREATE_DONT_BIND_BIT: c_int = 128;
pub const VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT: c_int = 256;
pub const VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT: c_int = 512;
pub const VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT: c_int = 65536;
pub const VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT: c_int = 131072;
pub const VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT: c_int = 262144;
pub const VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT: c_int = 65536;
pub const VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT: c_int = 262144;
pub const VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT: c_int = 131072;
pub const VMA_ALLOCATION_CREATE_STRATEGY_MASK: c_int = 458752;
pub const VmaAllocationCreateFlags = VkFlags;

pub const VMA_MEMORY_USAGE_UNKNOWN: c_int = 0;
pub const VMA_MEMORY_USAGE_GPU_ONLY: c_int = 1;
pub const VMA_MEMORY_USAGE_CPU_ONLY: c_int = 2;
pub const VMA_MEMORY_USAGE_CPU_TO_GPU: c_int = 3;
pub const VMA_MEMORY_USAGE_GPU_TO_CPU: c_int = 4;
pub const VMA_MEMORY_USAGE_CPU_COPY: c_int = 5;
pub const VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED: c_int = 6;
pub const VmaMemoryUsage = c_uint;
