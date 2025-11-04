import vulkan as vk
import time
import sys
import threading
import ctypes

def get_all_gpus():
    """Find all available GPU devices using Vulkan"""
    try:
        # Create Vulkan instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="GPU Keep-Alive",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )
        
        instance = vk.vkCreateInstance(create_info, None)
        
        # Get all physical devices (GPUs)
        physical_devices = vk.vkEnumeratePhysicalDevices(instance)
        
        gpus = []
        for device in physical_devices:
            props = vk.vkGetPhysicalDeviceProperties(device)
            # Only include GPU devices
            if props.deviceType in [vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 
                                    vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                    vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU]:
                gpus.append((instance, device, props))
        
        return gpus
    except Exception as e:
        print(f"Error initializing Vulkan: {e}")
        return []

def find_memory_type(physical_device, type_filter, properties):
    """Find suitable memory type for allocation"""
    mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    
    for i in range(mem_properties.memoryTypeCount):
        if (type_filter & (1 << i)) and \
           (mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
            return i
    
    return None

def keep_single_gpu_alive(instance, physical_device, props, gpu_index, interval=5):
    """
    Keep a single GPU alive by allocating and accessing VRAM
    """
    try:
        device_name = props.deviceName
        
        # Get queue family properties
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        
        # Find any queue family (compute or graphics)
        queue_family_index = None
        for i, family in enumerate(queue_families):
            if family.queueFlags & (vk.VK_QUEUE_COMPUTE_BIT | vk.VK_QUEUE_GRAPHICS_BIT):
                queue_family_index = i
                break
        
        if queue_family_index is None:
            print(f"GPU #{gpu_index} ({device_name}) - No suitable queue found, skipping")
            return
        
        # Create logical device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info]
        )
        
        logical_device = vk.vkCreateDevice(physical_device, device_create_info, None)
        queue = vk.vkGetDeviceQueue(logical_device, queue_family_index, 0)
        
        # Allocate minimal VRAM buffer (1MB = 1048576 bytes)
        buffer_size = 1048576
        
        buffer_create_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=buffer_size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(logical_device, buffer_create_info, None)
        
        # Get memory requirements
        mem_requirements = vk.vkGetBufferMemoryRequirements(logical_device, buffer)
        
        # Find device-local memory (VRAM)
        memory_type_index = find_memory_type(
            physical_device,
            mem_requirements.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        
        if memory_type_index is None:
            print(f"GPU #{gpu_index} ({device_name}) - No suitable VRAM found, skipping")
            return
        
        # Allocate VRAM
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=memory_type_index
        )
        
        device_memory = vk.vkAllocateMemory(logical_device, alloc_info, None)
        
        # Bind buffer to memory
        vk.vkBindBufferMemory(logical_device, buffer, device_memory, 0)
        
        # Create command pool
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        
        command_pool = vk.vkCreateCommandPool(logical_device, pool_info, None)
        
        # Allocate command buffer
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        command_buffers = vk.vkAllocateCommandBuffers(logical_device, alloc_info)
        command_buffer = command_buffers[0]
        
        print(f"GPU #{gpu_index} Keep-Alive started: {device_name}")
        print(f"GPU #{gpu_index} VRAM allocated: 1 MB")
        
        iteration = 0
        while True:
            # Record command to fill buffer (touches VRAM)
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            )
            
            vk.vkBeginCommandBuffer(command_buffer, begin_info)
            # Fill buffer with data - this actually accesses VRAM
            vk.vkCmdFillBuffer(command_buffer, buffer, 0, buffer_size, 0x00000000)
            vk.vkEndCommandBuffer(command_buffer)
            
            # Submit to queue - actually performs VRAM access
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[command_buffer]
            )
            
            vk.vkQueueSubmit(queue, 1, [submit_info], None)
            vk.vkQueueWaitIdle(queue)
            
            iteration += 1
            
            time.sleep(interval)
            
    except Exception as e:
        print(f"GPU #{gpu_index} error: {e}")

def keep_all_gpus_alive(interval=5):
    """
    Keep all detected GPUs alive with minimal VRAM access operations
    interval: seconds between operations (default 5)
    """
    try:
        # Find all GPUs
        gpus = get_all_gpus()
        
        if not gpus:
            print("No GPU devices found!")
            print("\nMake sure you have:")
            print("  1. Vulkan runtime installed (comes with GPU drivers)")
            print("  2. PyVulkan installed: pip install vulkan")
            sys.exit(1)
        
        print("=== Universal GPU Keep-Alive Script (Vulkan) ===")
        print(f"Found {len(gpus)} GPU(s):\n")
        
        for i, (instance, device, props) in enumerate(gpus):
            device_type = {
                vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "Discrete",
                vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "Integrated",
                vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: "Virtual"
            }.get(props.deviceType, "Unknown")
            
            print(f"  GPU #{i}: {props.deviceName} ({device_type})")
        
        print(f"\nRunning VRAM access operations every {interval} seconds on all GPUs")
        print(f"VRAM allocated per GPU: 1 MB")
        print(f"Keeps both GPU core and VRAM active")
        print("Press Ctrl+C to stop\n")
        
        # Create a thread for each GPU
        threads = []
        for i, (instance, device, props) in enumerate(gpus):
            thread = threading.Thread(
                target=keep_single_gpu_alive,
                args=(instance, device, props, i, interval),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            time.sleep(0.1)  # Small delay between GPU initializations
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nMulti-GPU Keep-Alive stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have Vulkan support installed:")
        print("  pip install vulkan")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Universal GPU Keep-Alive Script (Vulkan) ===")
    print("This script keeps GPU core AND VRAM active on ALL GPUs")
    print("Works with ANY GPU: NVIDIA, AMD, Intel Arc, etc.\n")
    
    # You can adjust the interval here (in seconds)
    keep_all_gpus_alive(interval=1)
