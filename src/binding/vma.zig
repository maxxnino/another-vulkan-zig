const vk = @import("vulkan");

pub const AllocatorCreateFlags = packed struct {
    /// \brief Allocator and all objects created from it will not be synchronized internally, so you must guarantee they are used from only one thread at a time or synchronized externally by you.
    ///
    /// Using this flag may increase performance because internal mutexes are not used.
    externally_synchronized: bool = false,

    /// \brief Enables usage of vk.KHR_dedicated_allocation extension.
    ///
    /// The flag works only if AllocatorCreateInfo::vulkanApiVersion `== vk.API_VERSION_1_0`.
    /// When it's `vk.API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.
    ///
    /// Using this extenion will automatically allocate dedicated blocks of memory for
    /// some buffers and images instead of suballocating place for them out of bigger
    /// memory blocks (as if you explicitly used #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    /// flag) when it is recommended by the driver. It may improve performance on some
    /// GPUs.
    ///
    /// You may set this flag only if you found out that following device extensions are
    /// supported, you enabled them while creating Vulkan device passed as
    /// AllocatorCreateInfo::device, and you want them to be used internally by this
    /// library:
    ///
    /// - vk.KHR_get_memory_requirements2 (device extension)
    /// - vk.KHR_dedicated_allocation (device extension)
    ///
    /// When this flag is set, you can experience following warnings reported by Vulkan
    /// validation layer. You can ignore them.
    ///
    /// > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
    dedicated_allocation_khr: bool = false,

    /// Enables usage of vk.KHR_bind_memory2 extension.
    ///
    /// The flag works only if AllocatorCreateInfo::vulkanApiVersion `== vk.API_VERSION_1_0`.
    /// When it's `vk.API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.
    ///
    /// You may set this flag only if you found out that this device extension is supported,
    /// you enabled it while creating Vulkan device passed as AllocatorCreateInfo::device,
    /// and you want it to be used internally by this library.
    ///
    /// The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
    /// which allow to pass a chain of `pNext` structures while binding.
    /// This flag is required if you use `pNext` parameter in BindBufferMemory2() or BindImageMemory2().
    bind_memory2_khr: bool = false,

    /// Enables usage of vk.EXT_memory_budget extension.
    ///
    /// You may set this flag only if you found out that this device extension is supported,
    /// you enabled it while creating Vulkan device passed as AllocatorCreateInfo::device,
    /// and you want it to be used internally by this library, along with another instance extension
    /// vk.KHR_get_physical_device_properties2, which is required by it (or Vulkan 1.1, where this extension is promoted).
    ///
    /// The extension provides query for current memory usage and budget, which will probably
    /// be more accurate than an estimation used by the library otherwise.
    memory_budget_ext: bool = false,

    amd_device_coherent_memory: bool = false,
    buffer_device_address: bool = false,
    ext_memory_priority: bool = false,
    __reserved_bits_07_31: u25 = 0,

    pub usingnamespace vk.FlagsMixin(@This(), vk.Flags);
};
pub const RECORD_FLUSH_AFTER_CALL_BIT: c_int = 1;
pub const RECORD_FLAG_BITS_MAX_ENUM: c_int = 2147483647;
pub const enum_RecordFlagBits = c_uint;
pub const RecordFlagBits = enum_RecordFlagBits;
pub const RecordFlags = vk.Flags;

/// # Simple patterns
/// ## Render targets
/// - When: Any resources that you frequently write and read on GPU, e.g.
/// images used as color attachments (aka "render targets"), depth-stencil attachments,
/// images/buffers used as storage image/buffer (aka "Unordered Access View (UAV)").
/// - What to do: Create them in video memory that is fastest to access from GPU using VMA_MEMORY_USAGE_GPU_ONLY.
/// Consider using VK_KHR_dedicated_allocation extension and/or manually creating them as dedicated allocations
/// using VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, especially if they are large or if you plan to destroy
/// and recreate them e.g. when display resolution changes. Prefer to create such resources first and all
/// other GPU resources (like textures and vertex buffers) later.
///
/// ## Immutable resources
/// - When: Any resources that you fill on CPU only once (aka "immutable") or infrequently and
/// then read frequently on GPU, e.g. textures, vertex and index buffers, constant buffers that don't change often.
/// - What to do: Create them in video memory that is fastest to access from GPU using VMA_MEMORY_USAGE_GPU_ONLY.
/// To initialize content of such resource, create a CPU-side (aka "staging") copy of it in system memory
/// - VMA_MEMORY_USAGE_CPU_ONLY, map it, fill it, and submit a transfer from it to the GPU resource.
/// You can keep the staging copy if you need it for another upload transfer in the future.
/// If you don't, you can destroy it or reuse this buffer for uploading different resource after the transfer finishes.
/// Prefer to create just buffers in system memory rather than images, even for uploading textures.
/// Use vkCmdCopyBufferToImage(). Dont use images with VK_IMAGE_TILING_LINEAR.
///
/// ## Dynamic resources
/// - When: Any resources that change frequently (aka "dynamic"),
/// e.g. every frame or every draw call, written on CPU, read on GPU.
/// - What to do: Create them using VMA_MEMORY_USAGE_CPU_TO_GPU.
/// You can map it and write to it directly on CPU, as well as read from it on GPU.
/// This is a more complex situation. Different solutions are possible, and the best one depends on specific GPU type,
/// but you can use this simple approach for the start. Prefer to write to such resource sequentially (e.g. using memcpy).
/// Don't perform random access or any reads from it on CPU, as it may be very slow.
/// Also note that textures written directly from the host through a mapped pointer need to be in LINEAR not OPTIMAL layout.
///
/// ## Readback
/// - When: Resources that contain data written by GPU that you want to read back on CPU, e.g. results of some computations.
/// - What to do: Create them using VMA_MEMORY_USAGE_GPU_TO_CPU.
/// You can write to them directly on GPU, as well as map and read them on CPU.
/// # Advanced patterns
/// ## Detecting integrated graphics
/// - You can support integrated graphics (like Intel HD Graphics, AMD APU) better by detecting it in Vulkan.
/// To do it, call vkGetPhysicalDeviceProperties(), inspect VkPhysicalDeviceProperties::deviceType
/// and look for VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU. When you find it, you can assume that memory is unified
/// and all memory types are comparably fast to access from GPU, regardless of VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.
/// - You can then sum up sizes of all available memory heaps and treat them as useful for your GPU resources,
/// instead of only DEVICE_LOCAL ones. You can also prefer to create your resources in memory types
/// that are HOST_VISIBLE to map them directly instead of submitting explicit transfer (see below).
///
/// ## Direct access versus transfer
/// - For resources that you frequently write on CPU and read on GPU, many solutions are possible:
/// - Create one copy in video memory using VMA_MEMORY_USAGE_GPU_ONLY, second copy in system memory using
/// VMA_MEMORY_USAGE_CPU_ONLY and submit explicit transfer each time.
/// - Create just a single copy using VMA_MEMORY_USAGE_CPU_TO_GPU, map it and fill it on CPU, read it directly on GPU.
/// - Create just a single copy using VMA_MEMORY_USAGE_CPU_ONLY, map it and fill it on CPU, read it directly on GPU.
/// - Which solution is the most efficient depends on your resource and especially on the GPU.
/// It is best to measure it and then make the decision. Some general recommendations:
/// - On integrated graphics use (2) or (3) to avoid unnecessary time
/// and memory overhead related to using a second copy and making transfer.
/// - For small resources (e.g. constant buffers) use (2). Discrete AMD cards have special 256 MiB pool of video memory
/// that is directly mappable. Even if the resource ends up in system memory,
/// its data may be cached on GPU after first fetch over PCIe bus.
/// - For larger resources (e.g. textures), decide between (1) and (2).
/// You may want to differentiate NVIDIA and AMD, e.g. by looking for memory type that is
/// both DEVICE_LOCAL and HOST_VISIBLE. When you find it, use (2), otherwise use (1).
/// - Similarly, for resources that you frequently write on GPU and read on CPU, multiple solutions are possible:
///
/// - Create one copy in video memory using VMA_MEMORY_USAGE_GPU_ONLY, second copy in system memory
/// using VMA_MEMORY_USAGE_GPU_TO_CPU and submit explicit tranfer each time.
/// - Create just single copy using VMA_MEMORY_USAGE_GPU_TO_CPU, write to it directly on GPU, map it and read it on CPU.
/// - You should take some measurements to decide which option is faster in case of your specific resource.
///
/// Note that textures accessed directly from the host through a mapped pointer need to be in LINEAR layout,
/// which may slow down their usage on the device. Textures accessed only by the device
/// and transfer operations can use OPTIMAL layout.
///
/// If you don't want to specialize your code for specific types of GPUs, you can still make
/// an simple optimization for cases when your resource ends up in mappable memory to use it directly
/// in this case instead of creating CPU-side staging copy. For details see Finding out if memory is mappable.0
pub const MemoryUsage = enum(i32) {
    /// No intended memory usage specified.
    /// Use other members of AllocationCreateInfo to specify your requirements.
    unknown = 0,
    /// Memory will be used on device only, so fast access from the device is preferred.
    /// It usually means device-local GPU (video) memory.
    /// No need to be mappable on host.
    /// It is roughly equivalent of `D3D12_HEAP_TYPE_DEFAULT`.
    ///
    /// Usage:
    ///
    /// - Resources written and read by device, e.g. images used as attachments.
    /// - Resources transferred from host once (immutable) or infrequently and read by
    /// device multiple times, e.g. textures to be sampled, vertex buffers, uniform
    /// (constant) buffers, and majority of other types of resources used on GPU.
    ///
    /// Allocation may still end up in `HOST_VISIBLE` memory on some implementations.
    /// In such case, you are free to map it.
    /// You can use #VMA_ALLOCATION_CREATE_MAPPED_BIT with this usage type.
    gpu_only = 1,
    /// Memory will be mappable on host.
    /// It usually means CPU (system) memory.
    /// Guarantees to be `HOST_VISIBLE` and `HOST_COHERENT`.
    /// CPU access is typically uncached. Writes may be write-combined.
    /// Resources created in this pool may still be accessible to the device, but access to them can be slow.
    /// It is roughly equivalent of `D3D12_HEAP_TYPE_UPLOAD`.
    ///
    /// Usage: Staging copy of resources used as transfer source.
    cpu_only = 2,
    /// Memory that is both mappable on host (guarantees to be `HOST_VISIBLE`) and preferably fast to access by GPU.
    /// CPU access is typically uncached. Writes may be write-combined.
    ///
    /// Usage: Resources written frequently by host (dynamic), read by device.
    /// E.g. textures, vertex buffers, uniform buffers updated every frame or every draw call.
    cpu_to_gpu = 3,
    /// Memory mappable on host (guarantees to be `HOST_VISIBLE`) and cached.
    /// It is roughly equivalent of `D3D12_HEAP_TYPE_READBACK`.
    ///
    /// Usage:
    ///
    /// - Resources written by device, read by host - results of some computations, e.g. screen capture, average scene luminance for HDR tone mapping.
    /// - Any resources read or accessed randomly on host, e.g. CPU-side copy of vertex buffer used as source of transfer, but also used for collision detection.
    gpu_to_cpu = 4,
    /// CPU memory - memory that is preferably not `DEVICE_LOCAL`, but also not guaranteed to be `HOST_VISIBLE`.
    ///
    /// Usage: Staging copy of resources moved from GPU memory to CPU memory as part
    /// of custom paging/residency mechanism, to be moved back to GPU memory when needed.
    cpu_copy = 5,
    /// Lazily allocated GPU memory having `vk.MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`.
    /// Exists mostly on mobile platforms. Using it on desktop PC or other GPUs with no such memory type present will fail the allocation.
    ///
    /// Usage: Memory for transient attachment images (color attachments, depth attachments etc.), created with `vk.IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.
    ///
    /// Allocations with this usage are always created as dedicated - it implies #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    gpu_lazily_allocated = 6,
    auto = 7,

    auto_prefer_device = 8,
    auto_prefer_host = 9,
};
/// Flags to be passed as AllocationCreateInfo::flags.
pub const AllocationCreateFlags = packed struct {
    /// \brief Set this flag if the allocation should have its own memory block.
    ///
    /// Use it for special, big resources, like fullscreen images used as attachments.
    ///
    /// You should not use this flag if AllocationCreateInfo::pool is not null.
    dedicated_memory: bool = false,

    /// \brief Set this flag to only try to allocate from existing `vk.DeviceMemory` blocks and never create new such block.
    ///
    /// If new allocation cannot be placed in any of the existing blocks, allocation
    /// fails with `error.VK_OUT_OF_DEVICE_MEMORY` error.
    ///
    /// You should not use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT and
    /// #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT at the same time. It makes no sense.
    ///
    /// If AllocationCreateInfo::pool is not null, this flag is implied and ignored. */
    never_allocate: bool = false,
    /// \brief Set this flag to use a memory that will be persistently mapped and retrieve pointer to it.
    ///
    /// Pointer to mapped memory will be returned through AllocationInfo::pMappedData.
    ///
    /// Is it valid to use this flag for allocation made from memory type that is not
    /// `HOST_VISIBLE`. This flag is then ignored and memory is not mapped. This is
    /// useful if you need an allocation that is efficient to use on GPU
    /// (`DEVICE_LOCAL`) and still want to map it directly if possible on platforms that
    /// support it (e.g. Intel GPU).
    ///
    /// You should not use this flag together with #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT.
    create_mapped: bool = false,
    /// Allocation created with this flag can become lost as a result of another
    /// allocation with #VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT flag, so you
    /// must check it before use.
    ///
    /// To check if allocation is not lost, call GetAllocationInfo() and check if
    /// AllocationInfo::deviceMemory is not `.Null`.
    ///
    /// For details about supporting lost allocations, see Lost Allocations
    /// chapter of User Guide on Main Page.
    ///
    /// You should not use this flag together with #VMA_ALLOCATION_CREATE_MAPPED_BIT.
    can_become_lost: bool = false,
    /// While creating allocation using this flag, other allocations that were
    /// created with flag #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT can become lost.
    ///
    /// For details about supporting lost allocations, see Lost Allocations
    /// chapter of User Guide on Main Page.
    can_make_other_lost: bool = false,
    /// Set this flag to treat AllocationCreateInfo::pUserData as pointer to a
    /// null-terminated string. Instead of copying pointer value, a local copy of the
    /// string is made and stored in allocation's `pUserData`. The string is automatically
    /// freed together with the allocation. It is also used in BuildStatsString().
    user_data_copy_string: bool = false,
    /// Allocation will be created from upper stack in a double stack pool.
    ///
    /// This flag is only allowed for custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT flag.
    upper_address: bool = false,
    /// Create both buffer/image and allocation, but don't bind them together.
    /// It is useful when you want to bind yourself to do some more advanced binding, e.g. using some extensions.
    /// The flag is meaningful only with functions that bind by default: CreateBuffer(), CreateImage().
    /// Otherwise it is ignored.
    dont_bind: bool = false,
    /// Create allocation only if additional device memory required for it, if any, won't exceed
    /// memory budget. Otherwise return `error.VK_OUT_OF_DEVICE_MEMORY`.
    within_budget: bool = false,

    __reserved_bits_09_15: u7 = 0,

    /// Allocation strategy that chooses smallest possible free range for the
    /// allocation.
    strategy_best_fit: bool = false,
    /// Allocation strategy that chooses biggest possible free range for the
    /// allocation.
    strategy_worst_fit: bool = false,
    /// Allocation strategy that chooses first suitable free range for the
    /// allocation.
    ///
    /// "First" doesn't necessarily means the one with smallest offset in memory,
    /// but rather the one that is easiest and fastest to find.
    strategy_first_fit: bool = false,

    __reserved_bits_19_31: u13 = 0,

    /// Allocation strategy that tries to minimize memory usage.
    pub const STRATEGY_MIN_MEMORY = AllocationCreateFlags{ .strategy_best_fit = true };
    /// Allocation strategy that tries to minimize allocation time.
    pub const STRATEGY_MIN_TIME = AllocationCreateFlags{ .strategy_first_fit = true };
    /// Allocation strategy that tries to minimize memory fragmentation.
    pub const STRATEGY_MIN_FRAGMENTATION = AllocationCreateFlags{ .strategy_worst_fit = true };

    /// A bit mask to extract only `STRATEGY` bits from entire set of flags.
    pub const STRATEGY_MASK = AllocationCreateFlags{
        .strategy_best_fit = true,
        .strategy_worst_fit = true,
        .strategy_first_fit = true,
    };

    pub usingnamespace vk.FlagsMixin(@This(), vk.Flags);
};
pub const POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT: c_int = 2;
pub const POOL_CREATE_LINEAR_ALGORITHM_BIT: c_int = 4;
pub const POOL_CREATE_BUDDY_ALGORITHM_BIT: c_int = 8;
pub const POOL_CREATE_ALGORITHM_MASK: c_int = 12;
pub const POOL_CREATE_FLAG_BITS_MAX_ENUM: c_int = 2147483647;
pub const enum_PoolCreateFlagBits = c_uint;
pub const PoolCreateFlagBits = enum_PoolCreateFlagBits;
pub const PoolCreateFlags = vk.Flags;
pub const DEFRAGMENTATION_FLAG_INCREMENTAL: c_int = 1;
pub const DEFRAGMENTATION_FLAG_BITS_MAX_ENUM: c_int = 2147483647;
pub const enum_DefragmentationFlagBits = c_uint;
pub const DefragmentationFlagBits = enum_DefragmentationFlagBits;
pub const DefragmentationFlags = vk.Flags;
pub const VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT: c_int = 1;
pub const VIRTUAL_BLOCK_CREATE_BUDDY_ALGORITHM_BIT: c_int = 2;
pub const VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK: c_int = 3;
pub const VIRTUAL_BLOCK_CREATE_FLAG_BITS_MAX_ENUM: c_int = 2147483647;
pub const enum_VirtualBlockCreateFlagBits = c_uint;
pub const VirtualBlockCreateFlagBits = enum_VirtualBlockCreateFlagBits;
pub const VirtualBlockCreateFlags = vk.Flags;
pub const VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT: c_int = 64;
pub const VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT: c_int = 65536;
pub const VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT: c_int = 262144;
pub const VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT: c_int = 131072;
pub const VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK: c_int = 458752;
pub const VIRTUAL_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM: c_int = 2147483647;
pub const enum_VirtualAllocationCreateFlagBits = c_uint;
pub const VirtualAllocationCreateFlagBits = enum_VirtualAllocationCreateFlagBits;
pub const VirtualAllocationCreateFlags = vk.Flags;
pub const Allocator = enum(usize) {
    null_handle = 0,
    _,

    /// Creates Allocator object.
    pub fn create(createInfo: AllocatorCreateInfo) !Allocator {
        var result: Allocator = undefined;
        const rc = vmaCreateAllocator(&createInfo, &result);
        if (@enumToInt(rc) >= 0) return result;

        return error.VMACreateFailed;
    }
    /// Destroys allocator object.
    pub fn destroy(self: Allocator) void {
        vmaDestroyAllocator(self);
    }
    /// @param[out] pBuffer Buffer that was created.
    /// @param[out] pAllocation Allocation that was created.
    /// @param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function GetAllocationInfo().
    ///
    /// This function automatically:
    ///
    /// -# Creates buffer.
    /// -# Allocates appropriate memory for it.
    /// -# Binds the buffer with the memory.
    ///
    /// If any of these operations fail, buffer and allocation are not created,
    /// returned value is negative error code, *pBuffer and *pAllocation are null.
    ///
    /// If the function succeeded, you must destroy both buffer and allocation when you
    /// no longer need them using either convenience function DestroyBuffer() or
    /// separately, using `vkDestroyBuffer()` and FreeMemory().
    ///
    /// If VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag was used,
    /// vk.KHR_dedicated_allocation extension is used internally to query driver whether
    /// it requires or prefers the new buffer to have dedicated allocation. If yes,
    /// and if dedicated allocation is possible (AllocationCreateInfo::pool is null
    /// and VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT is not used), it creates dedicated
    /// allocation for this buffer, just like when using
    /// VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    pub fn createBuffer(
        allocator: Allocator,
        bufferCreateInfo: vk.BufferCreateInfo,
        allocationCreateInfo: AllocationCreateInfo,
    ) !CreateBufferResult {
        return createBufferAndGetInfo(allocator, bufferCreateInfo, allocationCreateInfo, null);
    }
    pub fn createBufferAndGetInfo(
        allocator: Allocator,
        bufferCreateInfo: vk.BufferCreateInfo,
        allocationCreateInfo: AllocationCreateInfo,
        outInfo: ?*AllocationInfo,
    ) !CreateBufferResult {
        var result: CreateBufferResult = undefined;
        const rc = vmaCreateBuffer(
            allocator,
            &bufferCreateInfo,
            &allocationCreateInfo,
            &result.buffer,
            &result.allocation,
            outInfo,
        );
        if (@enumToInt(rc) >= 0) return result;
        return switch (rc) {
            .error_out_of_host_memory => error.out_of_host_memory,
            .error_out_of_device_memory => error.out_of_device_memory,
            .error_too_many_objects => error.too_many_objects,
            .error_invalid_external_handle => error.invalid_external_handle,
            .error_invalid_opaque_capture_address => error.invalid_opaque_capture_address,
            .error_memory_map_failed => error.memory_map_failed,
            .error_fragmented_pool => error.fragmented_pool,
            .error_out_of_pool_memory => error.out_of_pool_memory,
            else => error.undocumented_error,
        };
    }

    /// \brief Destroys Vulkan buffer and frees allocated memory.
    ///
    /// This is just a convenience function equivalent to:
    ///
    /// \code
    /// vkDestroyBuffer(device, buffer, allocationCallbacks);
    /// FreeMemory(allocator, allocation);
    /// \endcode
    ///
    /// It it safe to pass null as buffer and/or allocation.
    /// fn destroyBuffer(allocator: Allocator, buffer: vk.Buffer, allocation: Allocation) void
    pub const destroyBuffer = vmaDestroyBuffer;

    /// Function similar to CreateBuffer().
    pub fn createImage(
        allocator: Allocator,
        imageCreateInfo: vk.ImageCreateInfo,
        allocationCreateInfo: AllocationCreateInfo,
    ) !CreateImageResult {
        return createImageAndGetInfo(allocator, imageCreateInfo, allocationCreateInfo, null);
    }
    pub fn createImageAndGetInfo(
        allocator: Allocator,
        imageCreateInfo: vk.ImageCreateInfo,
        allocationCreateInfo: AllocationCreateInfo,
        outInfo: ?*AllocationInfo,
    ) !CreateImageResult {
        var result: CreateImageResult = undefined;
        const rc = vmaCreateImage(
            allocator,
            &imageCreateInfo,
            &allocationCreateInfo,
            &result.image,
            &result.allocation,
            outInfo,
        );
        if (@enumToInt(rc) >= 0) return result;
        return switch (rc) {
            .error_out_of_host_memory => error.out_of_host_memory,
            .error_out_of_device_memory => error.out_of_device_memory,
            .error_too_many_objects => error.too_many_objects,
            .error_invalid_external_handle => error.invalid_external_handle,
            .error_invalid_opaque_capture_address => error.invalid_opaque_capture_address,
            .error_memory_map_failed => error.memory_map_failed,
            .error_fragmented_pool => error.fragmented_pool,
            .error_out_of_pool_memory => error.out_of_pool_memory,
            else => error.undocumented_error,
        };
    }

    /// \brief Destroys Vulkan image and frees allocated memory.
    ///
    /// This is just a convenience function equivalent to:
    ///
    /// \code
    /// vkDestroyImage(device, image, allocationCallbacks);
    /// FreeMemory(allocator, allocation);
    /// \endcode
    ///
    /// It is safe to pass null as image and/or allocation.
    pub const destroyImage = vmaDestroyImage;

    /// \brief Maps memory represented by given allocation and returns pointer to it.
    ///
    /// Maps memory represented by given allocation to make it accessible to CPU code.
    /// When succeeded, `*ppData` contains pointer to first byte of this memory.
    /// If the allocation is part of bigger `vk.DeviceMemory` block, the pointer is
    /// correctly offseted to the beginning of region assigned to this particular
    /// allocation.
    ///
    /// Mapping is internally reference-counted and synchronized, so despite raw Vulkan
    /// function `vkMapMemory()` cannot be used to map same block of `vk.DeviceMemory`
    /// multiple times simultaneously, it is safe to call this function on allocations
    /// assigned to the same memory block. Actual Vulkan memory will be mapped on first
    /// mapping and unmapped on last unmapping.
    ///
    /// If the function succeeded, you must call UnmapMemory() to unmap the
    /// allocation when mapping is no longer needed or before freeing the allocation, at
    /// the latest.
    ///
    /// It also safe to call this function multiple times on the same allocation. You
    /// must call UnmapMemory() same number of times as you called MapMemory().
    ///
    /// It is also safe to call this function on allocation created with
    /// #VMA_ALLOCATION_CREATE_MAPPED_BIT flag. Its memory stays mapped all the time.
    /// You must still call UnmapMemory() same number of times as you called
    /// MapMemory(). You must not call UnmapMemory() additional time to free the
    /// "0-th" mapping made automatically due to #VMA_ALLOCATION_CREATE_MAPPED_BIT flag.
    ///
    /// This function fails when used on allocation made in memory type that is not
    /// `HOST_VISIBLE`.
    ///
    /// This function always fails when called for allocation that was created with
    /// #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag. Such allocations cannot be
    /// mapped.
    ///
    /// This function doesn't automatically flush or invalidate caches.
    /// If the allocation is made from a memory types that is not `HOST_COHERENT`,
    /// you also need to use InvalidateAllocation() / FlushAllocation(), as required by Vulkan specification.
    pub fn mapMemory(allocator: Allocator, allocation: Allocation, comptime T: type) ![*]T {
        var data: *anyopaque = undefined;
        const rc = vmaMapMemory(allocator, allocation, &data);
        if (@enumToInt(rc) >= 0) return @intToPtr([*]T, @ptrToInt(data));
        return switch (rc) {
            .error_out_of_host_memory => error.out_of_host_memory,
            .error_out_of_device_memory => error.out_of_device_memory,
            .error_memory_map_failed => error.memory_map_failed,
            else => error.undocumented_error,
        };
    }

    /// \brief Unmaps memory represented by given allocation, mapped previously using MapMemory().
    ///
    /// For details, see description of MapMemory().
    ///
    /// This function doesn't automatically flush or invalidate caches.
    /// If the allocation is made from a memory types that is not `HOST_COHERENT`,
    /// you also need to use InvalidateAllocation() / FlushAllocation(), as required by Vulkan specification.
    pub fn unmapMemory(self: Allocator, allocation: Allocation) void {
        vmaUnmapMemory(self, allocation);
    }

    /// \brief Flushes memory of given allocation.
    ///
    /// Calls `vkFlushMappedMemoryRanges()` for memory associated with given range of given allocation.
    /// It needs to be called after writing to a mapped memory for memory types that are not `HOST_COHERENT`.
    /// Unmap operation doesn't do that automatically.
    ///
    /// - `offset` must be relative to the beginning of allocation.
    /// - `size` can be `vk.WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
    /// - `offset` and `size` don't have to be aligned.
    /// They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
    /// - If `size` is 0, this call is ignored.
    /// - If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
    /// this call is ignored.
    ///
    /// Warning! `offset` and `size` are relative to the contents of given `allocation`.
    /// If you mean whole allocation, you can pass 0 and `vk.WHOLE_SIZE`, respectively.
    /// Do not pass allocation's offset as `offset`!!!
    pub fn flushAllocation(self: Allocator, allocation: Allocation, offset: vk.DeviceSize, size: vk.DeviceSize) !void {
        const result = vmaFlushAllocation(self, allocation, offset, size);
        switch (result) {
            .success => {},
            .error_out_of_host_memory => return error.OutOfHostMemory,
            .error_out_of_device_memory => return error.OutOfDeviceMemory,
            else => return error.Unknown,
        }
    }
};
pub const CreateImageResult = struct {
    image: vk.Image,
    allocation: Allocation,
};

pub const CreateBufferResult = struct {
    buffer: vk.Buffer,
    allocation: Allocation,
};
pub const Pool = enum(usize) {
    null_handle = 0,
    _,
};
pub const Allocation = enum(usize) {
    null_handle = 0,
    _,
};
pub const DefragmentationContext = enum(usize) {
    null_handle = 0,
    _,
};
pub const VirtualBlock = enum(usize) {
    null_handle = 0,
    _,
};
pub const PFN_vmaAllocateDeviceMemoryFunction = ?fn (Allocator, u32, vk.DeviceMemory, vk.DeviceSize, ?*anyopaque) callconv(.C) void;
pub const PFN_vmaFreeDeviceMemoryFunction = ?fn (Allocator, u32, vk.DeviceMemory, vk.DeviceSize, ?*anyopaque) callconv(.C) void;
pub const DeviceMemoryCallbacks = extern struct {
    pfnAllocate: PFN_vmaAllocateDeviceMemoryFunction,
    pfnFree: PFN_vmaFreeDeviceMemoryFunction,
    pUserData: ?*anyopaque,
};
pub const VulkanFunctions = extern struct {
    getInstanceProcAddr: vk.PfnGetInstanceProcAddr,
    getDeviceProcAddr: vk.PfnGetDeviceProcAddr,
    getPhysicalDeviceProperties: vk.PfnGetPhysicalDeviceProperties,
    getPhysicalDeviceMemoryProperties: vk.PfnGetPhysicalDeviceMemoryProperties,
    allocateMemory: vk.PfnAllocateMemory,
    freeMemory: vk.PfnFreeMemory,
    mapMemory: vk.PfnMapMemory,
    unmapMemory: vk.PfnUnmapMemory,
    flushMappedMemoryRanges: vk.PfnFlushMappedMemoryRanges,
    invalidateMappedMemoryRanges: vk.PfnInvalidateMappedMemoryRanges,
    bindBufferMemory: vk.PfnBindBufferMemory,
    bindImageMemory: vk.PfnBindImageMemory,
    getBufferMemoryRequirements: vk.PfnGetBufferMemoryRequirements,
    getImageMemoryRequirements: vk.PfnGetImageMemoryRequirements,
    createBuffer: vk.PfnCreateBuffer,
    destroyBuffer: vk.PfnDestroyBuffer,
    createImage: vk.PfnCreateImage,
    destroyImage: vk.PfnDestroyImage,
    cmdCopyBuffer: vk.PfnCmdCopyBuffer,
    getBufferMemoryRequirements2: vk.PfnGetBufferMemoryRequirements2,
    getImageMemoryRequirements2: vk.PfnGetImageMemoryRequirements2,
    bindBufferMemory2: vk.PfnBindBufferMemory2,
    bindImageMemory2: vk.PfnBindImageMemory2,
    getPhysicalDeviceMemoryProperties2: vk.PfnGetPhysicalDeviceMemoryProperties2,
    getDeviceBufferMemoryRequirements: vk.PfnGetDeviceBufferMemoryRequirements,
    getDeviceImageMemoryRequirements: vk.PfnGetDeviceImageMemoryRequirements,
};
pub const RecordSettings = extern struct {
    flags: RecordFlags,
    pFilePath: [*:0]const u8,
};
pub const AllocatorCreateInfo = extern struct {
    flags: AllocatorCreateFlags = .{},

    /// Vulkan physical device.
    /// It must be valid throughout whole lifetime of created allocator. */
    physicalDevice: vk.PhysicalDevice,

    /// Vulkan device.
    /// It must be valid throughout whole lifetime of created allocator. */
    device: vk.Device,

    /// Preferred size of a single `VkDeviceMemory` block to be allocated from large heaps > 1 GiB. Optional.
    /// Set to 0 to use default, which is currently 256 MiB. */
    preferredLargeHeapBlockSize: vk.DeviceSize = 0,

    /// Custom CPU memory allocation callbacks. Optional.
    /// Optional, can be null. When specified, will also be used for all CPU-side memory allocations. */
    pAllocationCallbacks: ?*const vk.AllocationCallbacks = null,

    /// Informative callbacks for `vkAllocateMemory`, `vkFreeMemory`. Optional.
    /// Optional, can be null. */
    pDeviceMemoryCallbacks: ?*const DeviceMemoryCallbacks = null,

    /// \brief Either null or a pointer to an array of limits on maximum number of bytes that can be allocated out of particular Vulkan memory heap.
    /// If not NULL, it must be a pointer to an array of
    /// `VkPhysicalDeviceMemoryProperties::memoryHeapCount` elements, defining limit on
    /// maximum number of bytes that can be allocated out of particular Vulkan memory
    /// heap.
    /// Any of the elements may be equal to `VK_WHOLE_SIZE`, which means no limit on that
    /// heap. This is also the default in case of `pHeapSizeLimit` = NULL.
    /// If there is a limit defined for a heap:
    /// - If user tries to allocate more memory from that heap using this allocator,
    ///   the allocation fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    /// - If the limit is smaller than heap size reported in `VkMemoryHeap::size`, the
    ///   value of this limit will be reported instead when using vmaGetMemoryProperties().
    /// Warning! Using this feature may not be equivalent to installing a GPU with
    /// smaller amount of memory, because graphics driver doesn't necessary fail new
    /// allocations with `VK_ERROR_OUT_OF_DEVICE_MEMORY` result when memory capacity is
    /// exceeded. It may return success and just silently migrate some device memory
    /// blocks to system RAM. This driver behavior can also be controlled using
    /// VK_AMD_memory_overallocation_behavior extension.
    pHeapSizeLimit: ?[*]const vk.DeviceSize = null,

    /// \brief Pointers to Vulkan functions. Can be null.
    /// For details see [Pointers to Vulkan functions](@ref config_Vulkan_functions).
    pVulkanFunctions: ?*const VulkanFunctions = null,

    /// \brief Handle to Vulkan instance object.
    /// Starting from version 3.0.0 this member is no longer optional, it must be set!
    instance: vk.Instance,

    /// \brief Optional. The highest version of Vulkan that the application is designed to use.
    /// It must be a value in the format as created by macro `VK_MAKE_VERSION` or a constant like: `VK_API_VERSION_1_1`, `VK_API_VERSION_1_0`.
    /// The patch version number specified is ignored. Only the major and minor versions are considered.
    /// It must be less or equal (preferably equal) to value as passed to `vkCreateInstance` as `VkApplicationInfo::apiVersion`.
    /// Only versions 1.0, 1.1, 1.2 are supported by the current implementation.
    /// Leaving it initialized to zero is equivalent to `VK_API_VERSION_1_0`.
    vulkanApiVersion: u32,

    /// \brief Either null or a pointer to an array of external memory handle types for each Vulkan memory type.
    /// If not NULL, it must be a pointer to an array of `VkPhysicalDeviceMemoryProperties::memoryTypeCount`
    /// elements, defining external memory handle types of particular Vulkan memory type,
    /// to be passed using `VkExportMemoryAllocateInfoKHR`.
    /// Any of the elements may be equal to 0, which means not to use `VkExportMemoryAllocateInfoKHR` on this memory type.
    /// This is also the default in case of `pTypeExternalMemoryHandleTypes` = NULL.
    pTypeExternalMemoryHandleTypes: ?[*]const vk.ExternalMemoryHandleTypeFlagsKHR = null,
};
pub const AllocatorInfo = extern struct {
    instance: vk.Instance,
    physicalDevice: vk.PhysicalDevice,
    device: vk.Device,
};
pub const StatInfo = extern struct {
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
pub const Stats = extern struct {
    memoryType: [32]StatInfo,
    memoryHeap: [16]StatInfo,
    total: StatInfo,
};
pub const Budget = extern struct {
    blockBytes: vk.DeviceSize,
    allocationBytes: vk.DeviceSize,
    usage: vk.DeviceSize,
    budget: vk.DeviceSize,
};
pub const AllocationCreateInfo = extern struct {
    flags: AllocationCreateFlags = .{},

    /// \brief Intended usage of memory.
    /// You can leave #VMA_MEMORY_USAGE_UNKNOWN if you specify memory requirements in other way. \n
    /// If `pool` is not null, this member is ignored.
    usage: MemoryUsage = .unknown,

    /// \brief Flags that must be set in a Memory Type chosen for an allocation.
    /// Leave 0 if you specify memory requirements in other way. \n
    /// If `pool` is not null, this member is ignored.*/
    requiredFlags: vk.MemoryPropertyFlags = .{},

    /// \brief Flags that preferably should be set in a memory type chosen for an allocation.
    /// Set to 0 if no additional flags are preferred. \n
    /// If `pool` is not null, this member is ignored. */
    preferredFlags: vk.MemoryPropertyFlags = .{},

    /// \brief Bitmask containing one bit set for every memory type acceptable for this allocation.
    /// Value 0 is equivalent to `UINT32_MAX` - it means any memory type is accepted if
    /// it meets other requirements specified by this structure, with no further
    /// restrictions on memory type index. \n
    /// If `pool` is not null, this member is ignored.
    memoryTypeBits: u32 = 0,

    /// \brief Pool that this allocation should be created in.
    /// Leave `VK_NULL_HANDLE` to allocate from default pool. If not null, members:
    /// `usage`, `requiredFlags`, `preferredFlags`, `memoryTypeBits` are ignored.
    pool: Pool = .null_handle,

    /// \brief Custom general-purpose pointer that will be stored in #VmaAllocation, can be read as VmaAllocationInfo::pUserData and changed using vmaSetAllocationUserData().
    /// If #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT is used, it must be either
    /// null or pointer to a null-terminated string. The string will be then copied to
    /// internal buffer, so it doesn't need to be valid after allocation call.
    pUserData: ?*anyopaque = null,

    /// \brief A floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.
    /// It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object
    /// and this allocation ends up as dedicated or is explicitly forced as dedicated using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    /// Otherwise, it has the priority of a memory block where it is placed and this variable is ignored.
    priority: f32 = 1,
};
pub const PoolCreateInfo = extern struct {
    memoryTypeIndex: u32,
    flags: PoolCreateFlags,
    blockSize: vk.DeviceSize,
    minBlockCount: usize,
    maxBlockCount: usize,
    frameInUseCount: u32,
    priority: f32,
    minAllocationAlignment: vk.DeviceSize,
    pMemoryAllocateNext: ?*anyopaque,
};
pub const PoolStats = extern struct {
    size: vk.DeviceSize,
    unusedSize: vk.DeviceSize,
    allocationCount: usize,
    unusedRangeCount: usize,
    unusedRangeSizeMax: vk.DeviceSize,
    blockCount: usize,
};
pub const AllocationInfo = extern struct {
    memoryType: u32,
    deviceMemory: vk.DeviceMemory,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
    pMappedData: ?*anyopaque,
    pUserData: ?*anyopaque,
};
pub const DefragmentationInfo2 = extern struct {
    flags: DefragmentationFlags,
    allocationCount: u32,
    pAllocations: [*c]const Allocation,
    pAllocationsChanged: [*c]vk.Bool32,
    poolCount: u32,
    pPools: [*c]const Pool,
    maxCpuBytesToMove: vk.DeviceSize,
    maxCpuAllocationsToMove: u32,
    maxGpuBytesToMove: vk.DeviceSize,
    maxGpuAllocationsToMove: u32,
    commandBuffer: vk.CommandBuffer,
};
pub const DefragmentationPassMoveInfo = extern struct {
    allocation: Allocation,
    memory: vk.DeviceMemory,
    offset: vk.DeviceSize,
};
pub const DefragmentationPassInfo = extern struct {
    moveCount: u32,
    pMoves: [*c]DefragmentationPassMoveInfo,
};
pub const DefragmentationInfo = extern struct {
    maxBytesToMove: vk.DeviceSize,
    maxAllocationsToMove: u32,
};
pub const DefragmentationStats = extern struct {
    bytesMoved: vk.DeviceSize,
    bytesFreed: vk.DeviceSize,
    allocationsMoved: u32,
    deviceMemoryBlocksFreed: u32,
};
pub const VirtualBlockCreateInfo = extern struct {
    size: vk.DeviceSize,
    flags: VirtualBlockCreateFlagBits,
    pAllocationCallbacks: [*c]const vk.AllocationCallbacks,
};
pub const VirtualAllocationCreateInfo = extern struct {
    size: vk.DeviceSize,
    alignment: vk.DeviceSize,
    flags: VirtualAllocationCreateFlags,
    pUserData: ?*anyopaque,
};
pub const VirtualAllocationInfo = extern struct {
    size: vk.DeviceSize,
    pUserData: ?*anyopaque,
};
pub extern fn vmaCreateAllocator(pCreateInfo: *const AllocatorCreateInfo, pAllocator: *Allocator) vk.Result;
pub extern fn vmaDestroyAllocator(allocator: Allocator) void;
pub extern fn vmaGetAllocatorInfo(allocator: Allocator, pAllocatorInfo: [*c]AllocatorInfo) void;
pub extern fn vmaGetPhysicalDeviceProperties(allocator: Allocator, ppPhysicalDeviceProperties: [*c][*c]const vk.PhysicalDeviceProperties) void;
pub extern fn vmaGetMemoryProperties(allocator: Allocator, ppPhysicalDeviceMemoryProperties: [*c][*c]const vk.PhysicalDeviceMemoryProperties) void;
pub extern fn vmaGetMemoryTypeProperties(allocator: Allocator, memoryTypeIndex: u32, pFlags: [*c]vk.MemoryPropertyFlags) void;
pub extern fn vmaSetCurrentFrameIndex(allocator: Allocator, frameIndex: u32) void;
pub extern fn vmaCalculateStats(allocator: Allocator, pStats: [*c]Stats) void;
pub extern fn vmaGetHeapBudgets(allocator: Allocator, pBudgets: [*c]Budget) void;
pub extern fn vmaFindMemoryTypeIndex(allocator: Allocator, memoryTypeBits: u32, pAllocationCreateInfo: [*c]const AllocationCreateInfo, pMemoryTypeIndex: [*c]u32) vk.Result;
pub extern fn vmaFindMemoryTypeIndexForBufferInfo(allocator: Allocator, pBufferCreateInfo: [*c]const vk.BufferCreateInfo, pAllocationCreateInfo: [*c]const AllocationCreateInfo, pMemoryTypeIndex: [*c]u32) vk.Result;
pub extern fn vmaFindMemoryTypeIndexForImageInfo(allocator: Allocator, pImageCreateInfo: [*c]const vk.ImageCreateInfo, pAllocationCreateInfo: [*c]const AllocationCreateInfo, pMemoryTypeIndex: [*c]u32) vk.Result;
pub extern fn vmaCreatePool(allocator: Allocator, pCreateInfo: [*c]const PoolCreateInfo, pPool: [*c]Pool) vk.Result;
pub extern fn vmaDestroyPool(allocator: Allocator, pool: Pool) void;
pub extern fn vmaGetPoolStats(allocator: Allocator, pool: Pool, pPoolStats: [*c]PoolStats) void;
pub extern fn vmaMakePoolAllocationsLost(allocator: Allocator, pool: Pool, pLostAllocationCount: [*c]usize) void;
pub extern fn vmaCheckPoolCorruption(allocator: Allocator, pool: Pool) vk.Result;
pub extern fn vmaGetPoolName(allocator: Allocator, pool: Pool, ppName: [*c][*c]const u8) void;
pub extern fn vmaSetPoolName(allocator: Allocator, pool: Pool, pName: [*c]const u8) void;
pub extern fn vmaAllocateMemory(allocator: Allocator, pVkMemoryRequirements: [*c]const vk.MemoryRequirements, pCreateInfo: [*c]const AllocationCreateInfo, pAllocation: [*c]Allocation, pAllocationInfo: [*c]AllocationInfo) vk.Result;
pub extern fn vmaAllocateMemoryPages(allocator: Allocator, pVkMemoryRequirements: [*c]const vk.MemoryRequirements, pCreateInfo: [*c]const AllocationCreateInfo, allocationCount: usize, pAllocations: [*c]Allocation, pAllocationInfo: [*c]AllocationInfo) vk.Result;
pub extern fn vmaAllocateMemoryForBuffer(allocator: Allocator, buffer: vk.Buffer, pCreateInfo: [*c]const AllocationCreateInfo, pAllocation: [*c]Allocation, pAllocationInfo: [*c]AllocationInfo) vk.Result;
pub extern fn vmaAllocateMemoryForImage(allocator: Allocator, image: vk.Image, pCreateInfo: [*c]const AllocationCreateInfo, pAllocation: [*c]Allocation, pAllocationInfo: [*c]AllocationInfo) vk.Result;
pub extern fn vmaFreeMemory(allocator: Allocator, allocation: Allocation) void;
pub extern fn vmaFreeMemoryPages(allocator: Allocator, allocationCount: usize, pAllocations: [*c]const Allocation) void;
pub extern fn vmaGetAllocationInfo(allocator: Allocator, allocation: Allocation, pAllocationInfo: [*c]AllocationInfo) void;
pub extern fn vmaTouchAllocation(allocator: Allocator, allocation: Allocation) vk.Bool32;
pub extern fn vmaSetAllocationUserData(allocator: Allocator, allocation: Allocation, pUserData: ?*anyopaque) void;
pub extern fn vmaCreateLostAllocation(allocator: Allocator, pAllocation: [*c]Allocation) void;
pub extern fn vmaGetAllocationMemoryProperties(allocator: Allocator, allocation: Allocation, pFlags: [*c]vk.MemoryPropertyFlags) void;
pub extern fn vmaMapMemory(allocator: Allocator, allocation: Allocation, ppData: **anyopaque) vk.Result;
pub extern fn vmaUnmapMemory(allocator: Allocator, allocation: Allocation) void;
pub extern fn vmaFlushAllocation(allocator: Allocator, allocation: Allocation, offset: vk.DeviceSize, size: vk.DeviceSize) vk.Result;
pub extern fn vmaInvalidateAllocation(allocator: Allocator, allocation: Allocation, offset: vk.DeviceSize, size: vk.DeviceSize) vk.Result;
pub extern fn vmaFlushAllocations(allocator: Allocator, allocationCount: u32, allocations: [*]const Allocation, offsets: [*]const vk.DeviceSize, sizes: [*]const vk.DeviceSize) vk.Result;
pub extern fn vmaInvalidateAllocations(allocator: Allocator, allocationCount: u32, allocations: [*c]const Allocation, offsets: [*c]const vk.DeviceSize, sizes: [*c]const vk.DeviceSize) vk.Result;
pub extern fn vmaCheckCorruption(allocator: Allocator, memoryTypeBits: u32) vk.Result;
pub extern fn vmaDefragmentationBegin(allocator: Allocator, pInfo: [*c]const DefragmentationInfo2, pStats: [*c]DefragmentationStats, pContext: [*c]DefragmentationContext) vk.Result;
pub extern fn vmaDefragmentationEnd(allocator: Allocator, context: DefragmentationContext) vk.Result;
pub extern fn vmaBeginDefragmentationPass(allocator: Allocator, context: DefragmentationContext, pInfo: [*c]DefragmentationPassInfo) vk.Result;
pub extern fn vmaEndDefragmentationPass(allocator: Allocator, context: DefragmentationContext) vk.Result;
pub extern fn vmaDefragment(allocator: Allocator, pAllocations: [*c]const Allocation, allocationCount: usize, pAllocationsChanged: [*c]vk.Bool32, pDefragmentationInfo: [*c]const DefragmentationInfo, pDefragmentationStats: [*c]DefragmentationStats) vk.Result;
pub extern fn vmaBindBufferMemory(allocator: Allocator, allocation: Allocation, buffer: vk.Buffer) vk.Result;
pub extern fn vmaBindBufferMemory2(allocator: Allocator, allocation: Allocation, allocationLocalOffset: vk.DeviceSize, buffer: vk.Buffer, pNext: ?*const anyopaque) vk.Result;
pub extern fn vmaBindImageMemory(allocator: Allocator, allocation: Allocation, image: vk.Image) vk.Result;
pub extern fn vmaBindImageMemory2(allocator: Allocator, allocation: Allocation, allocationLocalOffset: vk.DeviceSize, image: vk.Image, pNext: ?*const anyopaque) vk.Result;
pub extern fn vmaCreateBuffer(allocator: Allocator, pBufferCreateInfo: *const vk.BufferCreateInfo, pAllocationCreateInfo: *const AllocationCreateInfo, pBuffer: *vk.Buffer, pAllocation: *Allocation, pAllocationInfo: ?*AllocationInfo) vk.Result;
pub extern fn vmaCreateBufferWithAlignment(allocator: Allocator, pBufferCreateInfo: [*c]const vk.BufferCreateInfo, pAllocationCreateInfo: [*c]const AllocationCreateInfo, minAlignment: vk.DeviceSize, pBuffer: [*c]vk.Buffer, pAllocation: [*c]Allocation, pAllocationInfo: [*c]AllocationInfo) vk.Result;
pub extern fn vmaDestroyBuffer(allocator: Allocator, buffer: vk.Buffer, allocation: Allocation) void;
pub extern fn vmaCreateImage(allocator: Allocator, pImageCreateInfo: *const vk.ImageCreateInfo, pAllocationCreateInfo: *const AllocationCreateInfo, pImage: *vk.Image, pAllocation: *Allocation, pAllocationInfo: ?*AllocationInfo) vk.Result;
pub extern fn vmaDestroyImage(allocator: Allocator, image: vk.Image, allocation: Allocation) void;
pub extern fn vmaCreateVirtualBlock(pCreateInfo: [*c]const VirtualBlockCreateInfo, pVirtualBlock: [*c]VirtualBlock) vk.Result;
pub extern fn vmaDestroyVirtualBlock(virtualBlock: VirtualBlock) void;
pub extern fn vmaIsVirtualBlockEmpty(virtualBlock: VirtualBlock) vk.Bool32;
pub extern fn vmaGetVirtualAllocationInfo(virtualBlock: VirtualBlock, offset: vk.DeviceSize, pVirtualAllocInfo: [*c]VirtualAllocationInfo) void;
pub extern fn vmaVirtualAllocate(virtualBlock: VirtualBlock, pCreateInfo: [*c]const VirtualAllocationCreateInfo, pOffset: [*c]vk.DeviceSize) vk.Result;
pub extern fn vmaVirtualFree(virtualBlock: VirtualBlock, offset: vk.DeviceSize) void;
pub extern fn vmaClearVirtualBlock(virtualBlock: VirtualBlock) void;
pub extern fn vmaSetVirtualAllocationUserData(virtualBlock: VirtualBlock, offset: vk.DeviceSize, pUserData: ?*anyopaque) void;
pub extern fn vmaCalculateVirtualBlockStats(virtualBlock: VirtualBlock, pStatInfo: [*c]StatInfo) void;
pub extern fn vmaBuildVirtualBlockStatsString(virtualBlock: VirtualBlock, ppStatsString: [*c][*c]u8, detailedMap: vk.Bool32) void;
pub extern fn vmaFreeVirtualBlockStatsString(virtualBlock: VirtualBlock, pStatsString: [*c]u8) void;
pub extern fn vmaBuildStatsString(allocator: Allocator, ppStatsString: [*c][*c]u8, detailedMap: vk.Bool32) void;
pub extern fn vmaFreeStatsString(allocator: Allocator, pStatsString: [*c]u8) void;
