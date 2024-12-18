// A bunch of Vulkan helper functions for the project

#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <optional>
#include <memory>

// ------------------ GLOBALS ------------------ //
extern VkDevice device;
extern VkQueue graphicsQueue;
extern VkPhysicalDevice physicalDevice;

// ------------------ FUNCTION POINTERS ------------------ //
namespace vkhfp {
#define FUNCTIONS \
    F(vkCreateAccelerationStructureKHR) \
    F(vkDestroyAccelerationStructureKHR) \
    F(vkGetAccelerationStructureBuildSizesKHR) \
    F(vkCmdBuildAccelerationStructuresKHR) \
    F(vkCmdWriteAccelerationStructuresPropertiesKHR) \
    F(vkCmdCopyAccelerationStructureKHR) \
    F(vkGetAccelerationStructureDeviceAddressKHR) \
    F(vkCmdPushDescriptorSetKHR) \
    F(vkCreateRayTracingPipelinesKHR) \
    F(vkGetRayTracingShaderGroupHandlesKHR) \
    F(vkCmdTraceRaysKHR)

#define F(name) inline PFN_##name name = nullptr;
    FUNCTIONS
#undef F

        template<typename T>
    void loadFunc(VkInstance instance, T& ptr, const char* name) {
        ptr = reinterpret_cast<T>(vkGetInstanceProcAddr(instance, name));
        if (!ptr) std::cerr << name << " isnt supported!\n";
    }

    inline void loadFuncPointers(VkInstance instance) {
#define F(name) loadFunc(instance, name, #name);
        FUNCTIONS
#undef F
    }
}

// ------------------ RAII WRAPPER ------------------ //
template<typename T, typename... Destroy>
struct VkhObject;

template<typename Object, typename... Destroy>
class VkhObj {
public:
    // constructors
    explicit VkhObj(Destroy... args) : objectP(std::make_shared<ObjWrapper>(VK_NULL_HANDLE, true, args...)) {}
    explicit VkhObj(Object obj, Destroy... args) : objectP(std::make_shared<ObjWrapper>(obj, true, args...)) {}

    // destructor
    ~VkhObj() = default;

    // copy constructor and assignment operator
    VkhObj(const VkhObj& other) : objectP(other.objectP) {}
    VkhObj& operator=(const VkhObj& other) {
        if (this != &other) {
            objectP = other.objectP;
        }
        return *this;
    }

    // move constructor and assignment operator
    VkhObj(VkhObj&& other) noexcept = default;
    VkhObj& operator=(VkhObj&& other) noexcept = default;

    // equality operators
    bool operator==(const VkhObj& other) const noexcept { return objectP->object == other.objectP->object; }
    bool operator!=(const VkhObj& other) const noexcept { return !(*this == other); }

    const Object& v() const noexcept { return objectP->object; }
    const Object* p() const noexcept { return &objectP->object; }
    Object& v() noexcept { return objectP->object; }
    Object* p() noexcept { return &objectP->object; }

    bool valid() const noexcept { return objectP->object != VK_NULL_HANDLE; }
    size_t use_count() const noexcept { return objectP.use_count(); }
    void setDestroy(bool destruction) { objectP->autoDestroy = destruction; }

    void reset(Destroy... d) {
        if (valid()) objectP->destroy();
        objectP = std::make_shared<ObjWrapper>(VK_NULL_HANDLE, true, d...);
    }
private:
    struct ObjWrapper {
        Object object;
        bool autoDestroy;
        std::tuple<Destroy...> destroyArgs;

        ObjWrapper(Object obj, bool destroy, Destroy... args) : object(obj), autoDestroy(destroy), destroyArgs(args...) {}

        void destroy() {
            std::apply([this](auto&&... args) { VkhObject<Object, Destroy...>::destroy(object, args...); }, destroyArgs);
            object = VK_NULL_HANDLE;
        }

        // destructor
        ~ObjWrapper() {
            if (autoDestroy && object != VK_NULL_HANDLE) {
                destroy();
            }
        }

        // disallow moving and copying
        ObjWrapper(const ObjWrapper&) = delete;
        ObjWrapper& operator=(const ObjWrapper&) = delete;
        ObjWrapper(ObjWrapper&&) = delete;
        ObjWrapper& operator=(ObjWrapper&&) = delete;
    };

    std::shared_ptr<ObjWrapper> objectP;
};

template<typename T>
struct std::hash<VkhObj<T>> {
    size_t operator()(const VkhObj<T>& obj) const noexcept {
        return (obj.valid()) ? (std::hash<T>()(obj.v())) : 0;
    }
};

template<>
struct VkhObject<VkCommandBuffer, VkCommandPool> {
    static void destroy(VkCommandBuffer commandBuffer, VkCommandPool commandPool) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
};

template<>
struct VkhObject<VkDescriptorSet, VkDescriptorPool> {
    static void destroy(VkDescriptorSet descriptorSet, VkDescriptorPool descriptorPool) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    }
};

#define MAKE_RAII(name, type, destruction) \
template<> \
struct VkhObject<type> { \
    static void destroy(type thing) { \
        destruction(device, thing, nullptr); \
    } \
}; \
using name = VkhObj<type>;

MAKE_RAII(VkhBuffer, VkBuffer, vkDestroyBuffer)
MAKE_RAII(VkhDeviceMemory, VkDeviceMemory, vkFreeMemory)

MAKE_RAII(VkhImage, VkImage, vkDestroyImage)
MAKE_RAII(VkhImageView, VkImageView, vkDestroyImageView)
MAKE_RAII(VkhSampler, VkSampler, vkDestroySampler)

MAKE_RAII(VkhCommandPool, VkCommandPool, vkDestroyCommandPool)
using VkhCommandBuffer = VkhObj<VkCommandBuffer, VkCommandPool>;

MAKE_RAII(VkhDescriptorPool, VkDescriptorPool, vkDestroyDescriptorPool)
MAKE_RAII(VkhDescriptorSetLayout, VkDescriptorSetLayout, vkDestroyDescriptorSetLayout)
using VkhDescriptorSet = VkhObj<VkDescriptorSet, VkDescriptorPool>;

MAKE_RAII(VkhPipeline, VkPipeline, vkDestroyPipeline)
MAKE_RAII(VkhPipelineLayout, VkPipelineLayout, vkDestroyPipelineLayout)
MAKE_RAII(VkhShaderModule, VkShaderModule, vkDestroyShaderModule)

MAKE_RAII(VkhRenderPass, VkRenderPass, vkDestroyRenderPass)
MAKE_RAII(VkhFramebuffer, VkFramebuffer, vkDestroyFramebuffer)

MAKE_RAII(VkhSemaphore, VkSemaphore, vkDestroySemaphore)
MAKE_RAII(VkhFence, VkFence, vkDestroyFence)
MAKE_RAII(VkhQueryPool, VkQueryPool, vkDestroyQueryPool)

MAKE_RAII(VkhSwapchainKHR, VkSwapchainKHR, vkDestroySwapchainKHR)

MAKE_RAII(VkhAccelerationStructure, VkAccelerationStructureKHR, vkhfp::vkDestroyAccelerationStructureKHR)

#undef MAKE_RAII

namespace vkh {
    typedef enum {
        BASE,
        NORMAL,
        METALLIC,
        EMISSIVE,
        OCCLUSION,
        DEPTH,
        CUBEMAP,
        ALPHA
    } TextureType;

    struct BufferObj {
        VkhBuffer buf{};
        VkhDeviceMemory mem{};

        void reset() {
            if (buf.valid()) buf.reset();
            if (mem.valid()) mem.reset();
        }
    };

    struct BufData {
        uint32_t vertexOffset = 0;
        uint32_t vertexCount = 0;
        uint32_t indexOffset = 0;
        uint32_t indexCount = 0;
    };

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        std::optional<uint32_t> computeFamily;
        std::optional<uint32_t> transferFamily;

        bool graphicsComplete() {
            return graphicsFamily.has_value();
        }

        bool presentComplete() {
            return presentFamily.has_value();
        }

        bool computeComplete() {
            return computeFamily.has_value();
        }

        bool transferComplete() {
            return transferFamily.has_value();
        }

        bool allComplete() {
            return graphicsComplete() && presentComplete() && computeComplete() && transferComplete();
        }
    };

    struct SCsupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    // ------------------ SWAP CHAIN ------------------ //
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) {
        VkExtent2D actualExtent = { width, height };

        // clamp the width and height to the min and max image extent
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent; //return the actual extent
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const VkSurfaceFormatKHR& format : availableFormats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }

        return availableFormats[0];

    }
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const VkPresentModeKHR& present : availablePresentModes) {
            if (present == VK_PRESENT_MODE_MAILBOX_KHR) {
                return present;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // gets the indices of each queue family (graphics, present, etc)
    QueueFamilyIndices findQueueFamilyIndices(VkSurfaceKHR surface, VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        // get the number of queue families and their properties
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        // iterate through all the queue families
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            const auto& family = queueFamilies[i];

            // check if the queue family supports graphics, compute and transfer operations
            if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
            if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) indices.computeFamily = i;
            if (family.queueFlags & VK_QUEUE_TRANSFER_BIT) indices.transferFamily = i;

            // check if the queue family supports presentation operations
            VkBool32 presSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presSupport);
            if (presSupport) indices.presentFamily = i;

            if (indices.allComplete()) {
                break;
            }
        }

        return indices;
    }

    // outputs details about what the swap chain supports
    SCsupportDetails querySCsupport(VkSurfaceKHR surface) {
        SCsupportDetails details;

        // get the surface capabilities of the physical device
        // example: the minimum and maximum number of images in the swap chain
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

        // get the number of supported surface formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

        // if the format count isnt 0, then get the actual surface format details
        // the surface format specifies the color space and pixel format
        if (formatCount) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
        }

        // get the number of supported present modes
        // present modes determine how the swapping of images to the display is handled
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

        // if any present modes are supported, get the present mode details
        if (presentModeCount) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }

    // ------------------ COMMAND BUFFERS ------------------ //
    VkhCommandPool createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags) {
        VkhCommandPool commandPool;
        VkCommandPoolCreateInfo poolInf{};
        poolInf.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInf.queueFamilyIndex = queueFamilyIndex; // the queue family that will be using this command pool
        poolInf.flags = createFlags;
        VkResult result = vkCreateCommandPool(device, &poolInf, nullptr, commandPool.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
        return commandPool;
    }

    VkhCommandBuffer allocateCommandBuffers(const VkhCommandPool& commandPool, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
        VkhCommandBuffer commandBuffer(commandPool.v());
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool.v();
        allocInfo.level = level;
        allocInfo.commandBufferCount = 1;

        VkResult result = vkAllocateCommandBuffers(device, &allocInfo, commandBuffer.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffer!!");
        }
        return commandBuffer;
    }

    VkhCommandBuffer beginSingleTimeCommands(const VkhCommandPool& commandPool) {
        VkhCommandBuffer commandBuffer = allocateCommandBuffers(commandPool);
        commandBuffer.setDestroy(false); // command buffer wont be autodestroyed
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; //one time command buffer
        vkBeginCommandBuffer(commandBuffer.v(), &beginInfo);
        return commandBuffer;
    }

    void endSingleTimeCommands(VkhCommandBuffer& commandBuffer, const VkhCommandPool& commandPool, VkQueue queue) {
        vkEndCommandBuffer(commandBuffer.v());
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = commandBuffer.p();
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); //submit the command buffer to the queue
        vkQueueWaitIdle(queue); //wait for the queue to be idle
        vkFreeCommandBuffers(device, commandPool.v(), 1, commandBuffer.p());
    }

    void createFB(const VkhRenderPass& renderPass, VkhFramebuffer& frameBuf, const VkImageView* attachments, size_t attatchmentCount, uint32_t width, uint32_t height) {
        if (frameBuf.valid()) frameBuf.reset();
        VkFramebufferCreateInfo frameBufferInfo{};
        frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        frameBufferInfo.renderPass = renderPass.v();
        frameBufferInfo.attachmentCount = static_cast<uint32_t>(attatchmentCount);
        frameBufferInfo.pAttachments = attachments;
        frameBufferInfo.width = width;
        frameBufferInfo.height = height;
        frameBufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, frameBuf.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    VkhSemaphore createSemaphore() {
        VkhSemaphore result{};

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, result.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphore!");
        }

        return result;
    }

    VkSubmitInfo createSubmitInfo(const VkCommandBuffer* commandBuffers, size_t commandBufferCount) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = nullptr;
        submitInfo.pWaitDstStageMask = nullptr;
        submitInfo.pCommandBuffers = commandBuffers;
        submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
        submitInfo.pSignalSemaphores = nullptr;
        submitInfo.signalSemaphoreCount = 0;
        return submitInfo;
    }

    VkSubmitInfo createSubmitInfo(const VkCommandBuffer* commandBuffers, size_t commandBufferCount, const VkPipelineStageFlags* waitStages, const VkhSemaphore& wait, const VkhSemaphore& signal) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = wait.p();
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.pCommandBuffers = commandBuffers;
        submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
        submitInfo.pSignalSemaphores = signal.p();
        submitInfo.signalSemaphoreCount = 1;
        return submitInfo;
    }

    VkSubmitInfo createSubmitInfo(const VkCommandBuffer* commandBuffers, size_t commandBufferCount, const VkPipelineStageFlags* waitStages, const VkSemaphore* wait, const VkSemaphore* signal, size_t waitSemaphoreCount, size_t signalSemaphoreCount) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphoreCount);
        submitInfo.pWaitSemaphores = wait;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.pCommandBuffers = commandBuffers;
        submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
        submitInfo.pSignalSemaphores = signal;
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphoreCount);
        return submitInfo;
    }

    // ------------------ MEMORY ------------------ //
    uint32_t findMemoryType(uint32_t tFilter, VkMemoryPropertyFlags prop) { //find the memory type based on the type filter and properties
        VkPhysicalDeviceMemoryProperties memP; //struct to hold memory properties
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memP); //get the memory properties for the physical device
        for (uint32_t i = 0; i < memP.memoryTypeCount; i++) { //loop through the memory types
            if ((tFilter & (1 << i)) && (memP.memoryTypes[i].propertyFlags & prop) == prop) { //if the memory type is suitable
                return i; //return the index of the memory type
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    VkDeviceAddress bufferDeviceAddress(const VkhBuffer& buffer) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer.v();

        return vkGetBufferDeviceAddress(device, &addrInfo);
    }

    VkDeviceAddress asDeviceAddress(const VkhAccelerationStructure& accelerationStructure) {
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = accelerationStructure.v();

        return vkhfp::vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);
    }

    template<typename ObjectT>
    void writeBuffer(const VkhDeviceMemory& bufferMem, const ObjectT* object, VkDeviceSize size) {
        if (object == nullptr) throw std::invalid_argument("Object is null!");
        if (size == 0) throw std::invalid_argument("Buffer size is 0!");

        void* data = nullptr;
        if (vkMapMemory(device, bufferMem.v(), 0, size, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map memory for buffer!");
        }

        if (data == nullptr) {
            throw std::runtime_error("Mapped memory is null!");
        }

        std::memcpy(data, object, static_cast<size_t>(size));
        vkUnmapMemory(device, bufferMem.v());
    }

    void copyBuffer(VkhBuffer& src, VkhBuffer& dst, const VkhCommandPool& commandPool, VkQueue queue, VkDeviceSize size) {
        VkhCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer.v(), src.v(), dst.v(), 1, &copyRegion);
        endSingleTimeCommands(commandBuffer, commandPool, queue);
    }

    void createBuffer(BufferObj& buffer, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags, VkMemoryAllocateFlags memAllocFlags = 0) {
        buffer.reset();

        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = usage;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        // create the buffer
        if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer.buf.p()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer!");
        }

        // get the memory requirements for the buffer
        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, buffer.buf.v(), &memoryRequirements);

        // mem allocation flags
        VkMemoryAllocateFlagsInfo allocFlagsInfo{};
        if (memAllocFlags) {
            allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            allocFlagsInfo.flags = memAllocFlags;
        }

        // allocate mem for the buffer
        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, memFlags);

        if (memAllocFlags) {
            allocateInfo.pNext = &allocFlagsInfo;
        }

        if (vkAllocateMemory(device, &allocateInfo, nullptr, buffer.mem.p()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate memory for the buffer!");
        }

        if (vkBindBufferMemory(device, buffer.buf.v(), buffer.mem.v(), 0) != VK_SUCCESS) {
            throw std::runtime_error("Failed to bind memory to buffer!");
        }
    }

    void createHostVisibleBuffer(BufferObj& buffer, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryAllocateFlags memAllocFlags = 0) {
        VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        createBuffer(buffer, size, usage, memFlags, memAllocFlags);
    }

    void createDeviceLocalBuffer(BufferObj& buffer, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryAllocateFlags memAllocFlags = 0) {
        VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        createBuffer(buffer, size, usage, memFlags, memAllocFlags);
    }

    template<typename ObjectT>
    void createAndWriteLocalBuffer(BufferObj& buffer, const ObjectT* data, VkDeviceSize size, const VkhCommandPool& commandPool, VkQueue queue, VkBufferUsageFlags usage, VkMemoryAllocateFlags memAllocFlags = 0) {
        createDeviceLocalBuffer(buffer, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memAllocFlags);

        BufferObj stagingBuffer;
        createHostVisibleBuffer(stagingBuffer, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memAllocFlags);

        // write the data to the staging buffer
        writeBuffer(stagingBuffer.mem, data, size);

        // copy the staging buffer to the device local buffer
        copyBuffer(stagingBuffer.buf, buffer.buf, commandPool, queue, size);
    }

    template<typename ObjectT>
    void createAndWriteHostBuffer(BufferObj& buffer, const ObjectT* data, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryAllocateFlags memAllocFlags = 0) {
        createHostVisibleBuffer(buffer, size, usage, memAllocFlags);
        writeBuffer(buffer.mem, data, size);
    }

    // ------------------ IMAGES ------------------ //
    VkFormat findDepthFormat() {
        //the formats that are supported
        std::vector<VkFormat> allowed = {
            VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT
        };

        for (VkFormat format : allowed) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props); //get the format properties

            // if the format has the depth stencil attachment bit
            if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                return format;
            }
        }
        throw std::runtime_error("failed to find suitable depth format!");
    }

    void transitionImageLayout(const VkhCommandBuffer& commandBuffer, const VkhImage& image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerCount, uint32_t levelCount, uint32_t baseMip) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.v();
        if (format == VK_FORMAT_D32_SFLOAT) { // if the format is a depth format
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        barrier.subresourceRange.baseMipLevel = baseMip;
        barrier.subresourceRange.levelCount = levelCount;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layerCount;

        // earliest stage in the pipeline that will wait on the barrier to be passed
        VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            }
            else {
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            }
        }
        else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }
            else {
                barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }
        }
        else if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            VkAccessFlags a = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            barrier.srcAccessMask = a;
            barrier.dstAccessMask = a;
            sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        else {
            throw std::invalid_argument("Unsupported layout transition!");
        }
        vkCmdPipelineBarrier(commandBuffer.v(), sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier); // insert the barrier into the command buffer
    }

    void transitionImageLayout(VkhCommandPool& commandPool, const VkhImage& image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerCount, uint32_t levelCount, uint32_t baseMip) {
        VkhCommandBuffer tempCommandBuffer = beginSingleTimeCommands(commandPool);
        transitionImageLayout(tempCommandBuffer, image, format, oldLayout, newLayout, layerCount, levelCount, baseMip);
        endSingleTimeCommands(tempCommandBuffer, commandPool, graphicsQueue);
    }

    VkFormat getTextureFormat(TextureType textureType) {
        switch (textureType) {
        case BASE:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case NORMAL:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case METALLIC:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case EMISSIVE:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case OCCLUSION:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case DEPTH:
            return VK_FORMAT_D32_SFLOAT;
        case CUBEMAP:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case ALPHA:
            return VK_FORMAT_R32_SFLOAT;
        }
    }

    void createImage(VkhImage& image, VkhDeviceMemory& imageMemory, uint32_t width, uint32_t height, VkFormat format, uint32_t mipLevels, uint32_t arrayLayers, bool cubeMap, VkImageUsageFlags usage, VkSampleCountFlagBits sample) {
        if (image.valid()) image.reset();
        if (imageMemory.valid()) imageMemory.reset();

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.arrayLayers = arrayLayers;
        imageInfo.mipLevels = mipLevels;
        imageInfo.format = format;

        // allows the gpu to format the image data in memory in the most efficient way
        // this means that the cpu cant easily read or write to the image though
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = sample;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (cubeMap) imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        if (vkCreateImage(device, &imageInfo, nullptr, image.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create color image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image.v(), &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        // memory is allocated on the gpu
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, imageMemory.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate color image memory!");
        }

        vkBindImageMemory(device, image.v(), imageMemory.v(), 0);
    }

    void createImage(VkhImage& image, VkhDeviceMemory& imageMemory, uint32_t width, uint32_t height, TextureType textureType, uint32_t mipLevels, uint32_t arrayLayers, bool cubeMap, VkImageUsageFlags usage, VkSampleCountFlagBits sample) {
        VkFormat format = getTextureFormat(textureType);
        createImage(image, imageMemory, width, height, format, mipLevels, arrayLayers, cubeMap, usage, sample);
    }

    void createSampler(VkhSampler& sampler, uint32_t mipLevels, TextureType type = BASE) {
        if (sampler.valid()) sampler.reset();
        VkSamplerCreateInfo samplerInf{};
        samplerInf.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInf.magFilter = VK_FILTER_LINEAR; // magnification filter
        samplerInf.minFilter = VK_FILTER_LINEAR; // minification filter

        // when the texture cords go out of bounds, repeat the texture
        samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        samplerInf.anisotropyEnable = VK_FALSE;
        samplerInf.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInf.unnormalizedCoordinates = VK_FALSE;
        switch (type) {
        case DEPTH:
            // when the texture cords go out of bounds, clamp the uv cords to the edge of the texture
            samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

            // instead of directly returning a sampled image, the sampler will compare a refrence value to the sampled value
            // this is particularly useful for shadowmapping
            samplerInf.compareEnable = VK_TRUE;
            samplerInf.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            break;
        case CUBEMAP:
            samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInf.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            break;
        }
        samplerInf.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInf.minLod = 0.0f;
        samplerInf.maxLod = static_cast<float>(mipLevels);
        if (vkCreateSampler(device, &samplerInf, nullptr, sampler.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    template<typename Texture>
    void createImageView(Texture& tex, TextureType type = BASE) {
        if (tex.imageView.valid()) tex.imageView.reset();

        VkImageViewCreateInfo viewInf{};
        viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInf.image = tex.image.v();
        viewInf.subresourceRange.baseArrayLayer = 0;
        viewInf.subresourceRange.layerCount = 1;
        viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInf.subresourceRange.aspectMask = (type == DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        viewInf.format = getTextureFormat(type);

        if (type == CUBEMAP) {
            viewInf.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            viewInf.subresourceRange.layerCount = 6;
        }

        viewInf.subresourceRange.baseMipLevel = 0;
        uint32_t level = (tex.mipLevels <= 0) ? 1 : tex.mipLevels;
        viewInf.subresourceRange.levelCount = level - viewInf.subresourceRange.baseMipLevel;
        if (vkCreateImageView(device, &viewInf, nullptr, tex.imageView.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }
    }

    template<typename Texture>
    void createImageView(Texture& tex, VkFormat format) {
        if (tex.imageView.valid()) tex.imageView.reset();

        VkImageViewCreateInfo viewInf{};
        viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInf.image = tex.image.v();
        viewInf.subresourceRange.baseArrayLayer = 0;
        viewInf.subresourceRange.layerCount = 1;
        viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInf.format = format;
        viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInf.subresourceRange.baseMipLevel = 0;

        uint32_t level = (tex.mipLevels <= 0) ? 1 : tex.mipLevels;
        viewInf.subresourceRange.levelCount = level - viewInf.subresourceRange.baseMipLevel;
        if (vkCreateImageView(device, &viewInf, nullptr, tex.imageView.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view! (swap)");
        }
    }

    // copy an image from one image to another
    void copyImage(VkhImage& srcImage, VkhImage& dstImage, VkImageLayout srcStart, VkImageLayout dstStart, VkImageLayout dstAfter, const VkhCommandBuffer& commandBuffer, VkFormat format, uint32_t width, uint32_t height, bool color) {
        transitionImageLayout(commandBuffer, srcImage, format, srcStart, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1, 0);
        transitionImageLayout(commandBuffer, dstImage, format, dstStart, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1, 0);

        VkImageCopy copy{};
        VkImageAspectFlagBits aspect = color ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
        copy.srcSubresource.aspectMask = aspect;
        copy.srcSubresource.mipLevel = 0;
        copy.srcSubresource.baseArrayLayer = 0;
        copy.srcSubresource.layerCount = 1;
        copy.dstSubresource.aspectMask = aspect;
        copy.dstSubresource.mipLevel = 0;
        copy.dstSubresource.baseArrayLayer = 0;
        copy.dstSubresource.layerCount = 1;
        copy.extent = { width, height, 1 };

        vkCmdCopyImage(commandBuffer.v(), srcImage.v(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        transitionImageLayout(commandBuffer, dstImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, dstAfter, 1, 1, 0);
        transitionImageLayout(commandBuffer, srcImage, format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);
    }

    void copyImage(VkhCommandPool& commandPool, VkhImage& srcImage, VkhImage& dstImage, VkImageLayout srcStart, VkImageLayout dstStart, VkImageLayout dstAfter, VkFormat format, uint32_t width, uint32_t height, bool color) {
        VkhCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);
        copyImage(srcImage, dstImage, srcStart, dstStart, dstAfter, commandBuffer, format, width, height, color);
        endSingleTimeCommands(commandBuffer, commandPool, graphicsQueue);
    }

    // ------------------ DESCRIPTOR SETS ------------------ //
    void createDSLayout(VkhDescriptorSetLayout& layout, const VkDescriptorSetLayoutBinding* bindings, size_t bindingCount, bool variableDescriptorCount, bool pushDescriptors) {
        if (layout.valid()) layout.reset();

        uint32_t count = static_cast<uint32_t>(bindingCount);

        std::vector<VkDescriptorBindingFlags> bindingFlags(count);
        if (variableDescriptorCount) {
            // set the last elem to be variable desc count
            bindingFlags.back() |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();
        bindingFlagsInfo.bindingCount = count;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.pBindings = bindings;
        layoutInfo.bindingCount = count;
        layoutInfo.pNext = &bindingFlagsInfo;
        if (pushDescriptors) layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, layout.p()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }

    void createDSPool(VkhDescriptorPool& pool, const VkDescriptorPoolSize* poolSizes, size_t poolSizeCount) {
        if (pool.valid()) pool.reset();

        uint32_t count = static_cast<uint32_t>(poolSizeCount);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.poolSizeCount = count;
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, pool.p()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
    }

    VkhDescriptorSet allocDS(VkhDescriptorSetLayout& layout, const VkhDescriptorPool& pool, uint32_t variableCount = 0) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorSetCount = 1;
        allocInfo.descriptorPool = pool.v();
        allocInfo.pSetLayouts = layout.p();

        VkDescriptorSetVariableDescriptorCountAllocateInfo varCountInfo{};
        if (variableCount) {
            varCountInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
            varCountInfo.descriptorSetCount = 1;
            varCountInfo.pDescriptorCounts = &variableCount;
            allocInfo.pNext = &varCountInfo;
        }

        VkhDescriptorSet set(pool.v());
        VkResult result = vkAllocateDescriptorSets(device, &allocInfo, set.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set!");
        }

        return set;
    }

    template<typename InfoType>
    VkWriteDescriptorSet createDSWrite(const VkhDescriptorSet& set, uint32_t binding, VkDescriptorType type, const InfoType* infos, size_t count) {
        // static assert if an invalid type is passed in
        static_assert(std::is_same_v<InfoType, VkDescriptorImageInfo>
            || std::is_same_v<InfoType, VkDescriptorBufferInfo>
            || std::is_same_v<InfoType, VkWriteDescriptorSetAccelerationStructureKHR>,
            "Invalid info type");

        VkWriteDescriptorSet d{};
        d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        d.dstSet = set.v();
        d.dstBinding = binding;
        d.descriptorType = type;
        d.descriptorCount = static_cast<uint32_t>(count);

        if constexpr (std::is_same_v<InfoType, VkDescriptorImageInfo>) { // if the info type is an image
            d.pImageInfo = infos;
        }
        else if constexpr (std::is_same_v<InfoType, VkDescriptorBufferInfo>) { // if the info type is a buffer
            d.pBufferInfo = infos;
        }
        else if constexpr (std::is_same_v<InfoType, VkWriteDescriptorSetAccelerationStructureKHR>) { // if the info type is an acceleration structure
            d.pNext = infos;
        }

        return d;
    }

    template<typename InfoType>
    VkWriteDescriptorSet createDSWrite(const VkhDescriptorSet& set, uint32_t binding, VkDescriptorType type, InfoType info) {
        // static assert if an invalid type is passed in
        static_assert(std::is_same_v<InfoType, VkDescriptorImageInfo>
            || std::is_same_v<InfoType, VkDescriptorBufferInfo>
            || std::is_same_v<InfoType, VkWriteDescriptorSetAccelerationStructureKHR>,
            "Invalid info type");

        VkWriteDescriptorSet d{};
        d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        d.dstSet = set.v();
        d.dstBinding = binding;
        d.descriptorType = type;
        d.descriptorCount = 1;

        if constexpr (std::is_same_v<InfoType, VkDescriptorImageInfo>) { // if the info type is an image
            d.pImageInfo = &info;
        }
        else if constexpr (std::is_same_v<InfoType, VkDescriptorBufferInfo>) { // if the info type is a buffer
            d.pBufferInfo = &info;
        }
        else if constexpr (std::is_same_v<InfoType, VkWriteDescriptorSetAccelerationStructureKHR>) { // if the info type is an acceleration structure
            d.pNext = &info;
        }

        return d;
    }

    VkDescriptorImageInfo createDSImageInfo(const VkhImageView& imageView, const VkhSampler& sampler, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        VkDescriptorImageInfo info{};
        info.imageLayout = layout;
        info.imageView = imageView.v();
        info.sampler = sampler.v();

        return info;
    }

    VkDescriptorSetLayoutBinding createDSLayoutBinding(uint32_t binding, size_t count, VkDescriptorType type, VkShaderStageFlags stageFlags) {
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorCount = static_cast<uint32_t>(count);
        layoutBinding.descriptorType = type;
        layoutBinding.stageFlags = stageFlags;

        return layoutBinding;
    }

    VkDescriptorPoolSize createDSPoolSize(size_t count, VkDescriptorType type) {
        VkDescriptorPoolSize poolSize{};
        poolSize.descriptorCount = static_cast<uint32_t>(count);
        poolSize.type = type;

        return poolSize;
    }

    // ------------------ PIPELINES ------------------ //
    VkhShaderModule createShaderModule(const std::vector<char>& code) { //takes in SPIRV binary and creates a shader module
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); //convert the char array to uint32_t array

        VkhShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, shaderModule.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }


    VkPipelineShaderStageCreateInfo createShaderStage(VkShaderStageFlagBits stage, const VkhShaderModule& shaderModule) {
        VkPipelineShaderStageCreateInfo s{};
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.stage = stage;
        s.module = shaderModule.v();
        s.pName = "main";
        return s;
    }

    VkVertexInputBindingDescription vertInputBindDesc(uint32_t binding, uint32_t stride, VkVertexInputRate inputRate) {
        VkVertexInputBindingDescription b{};
        b.binding = binding;
        b.stride = stride;
        b.inputRate = inputRate;

        return b;
    }

    VkVertexInputAttributeDescription vertInputAttrDesc(VkFormat format, uint32_t binding, uint32_t location, size_t offset) {
        VkVertexInputAttributeDescription a{};
        a.format = format;
        a.binding = binding;
        a.location = location;
        a.offset = static_cast<uint32_t>(offset);

        return a;
    }
};
