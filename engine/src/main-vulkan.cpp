#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define ENABLE_DEBUG

#include <stb_image_resize.h>

// headers
#include <dml.hpp>
#include <dvl.hpp>
#include <utils.hpp>
#include <vkhelper.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <future>
#include <optional>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <random>
#include <ctime>
#include <cmath>
#include <memory>

using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;

namespace cfg {
    constexpr uint32_t MAX_MODELS = 1600;
    constexpr uint32_t MAX_LIGHTS = 200;

    constexpr uint32_t SCREEN_WIDTH = 2560;
    constexpr uint32_t SCREEN_HEIGHT = 1600;

    const std::string ENGINE_VER = "v0.1.0";

    const std::string SHADER_DIR = "shaders/compiled/";
    const std::string MODEL_DIR = "assets/models/";
    const std::string SKYBOX_DIR = "assets/skyboxes/";
    const std::string FONT_DIR = "assets/fonts/";
}

struct MouseData {
    bool locked;

    float lastX;
    float lastY;

    float upAngle;
    float rightAngle;

    MouseData() :
        upAngle(0.0f),
        rightAngle(0.0f),
        lastX(0.0f),
        lastY(0.0f),
        locked(true)
    {
    }
};

// globals
MouseData mouse{};
VkQueue graphicsQueue{};
VkPhysicalDevice physicalDevice{};
GLFWwindow* window = nullptr;

VkSurfaceKHR surface{};
VkInstance instance{};
VkDevice device{};

class Engine {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
    }
private:
    struct CamData {
        dml::vec3 pos{ 0.0f, -0.75f, -3.5f };
        dml::vec4 quat{};

        dml::mat4 projectionMatrix{};
        dml::mat4 viewMatrix{};

        // buffers for the camera matrix ubo
        std::vector<VkhBuffer> buffers{};
        std::vector<VkhDeviceMemory> bufferMems{};

        float fov = 60.0f;
        float nearP = 0.01f;
        float farP = 100.0f;

        dml::mat4 getViewMatrix(MouseData& m) const {
            return dml::viewMatrix(pos, dml::radians(m.upAngle), dml::radians(m.rightAngle));
        }

        void updateQuaternion(MouseData& m) {
            dml::vec4 yRot = dml::angleAxis(dml::radians(m.upAngle), dml::vec3(1, 0, 0));
            dml::vec4 xRot = dml::angleAxis(dml::radians(m.rightAngle), dml::vec3(0, 1, 0));
            quat = yRot * xRot;
        }
    };

    struct LightDataObject {
        dml::vec3 pos{};
        dml::vec3 col{ 1.0f, 1.0f, 1.0f };
        dml::vec3 target{};

        dml::mat4 proj{};
        dml::mat4 view{};

        float intensity = 1.0f;
        float innerConeAngle = 0.23f;
        float outerConeAngle = 0.348f;
        float constantAttenuation = 1.0f;
        float linearAttenuation = 0.1f;
        float quadraticAttenuation = 0.032f;
    };

    struct Light {
        LightDataObject data{};

        std::vector<dvl::Texture> shadowMapData{};
        std::vector<VkhFramebuffer> frameBuffer{};

        bool followPlayer = false;
    };

    struct LightDataSSBO {
        LightDataObject lightCords[cfg::MAX_LIGHTS];
    };

    struct ModelInstance {
        dml::mat4 model{};
        uint32_t render = 1;
    };

    struct ModelInstanceData {
        ModelInstance object[cfg::MAX_MODELS];
    };
    struct CamUBO {
        dml::mat4 view{};
        dml::mat4 proj{};
    };

    struct ShadowMapDim {
        uint32_t width = 2048;
        uint32_t height = 2048;
    };

    struct SkyboxObject {
        dvl::Texture cubemap{};

        VkhPipelineLayout pipelineLayout{};
        VkhPipeline pipeline{};

        vkh::BufData bufferData{};
        VkhBuffer vertBuffer{};
        VkhDeviceMemory vertBufferMem{};
        VkhBuffer indBuffer{};
        VkhDeviceMemory indBufferMem{};

        float* imgData{ nullptr };

        std::vector<uint32_t> indices{
                0, 1, 2, 2, 3, 0,
                7, 6, 5, 5, 4, 7,
                4, 5, 1, 1, 0, 4,
                3, 2, 6, 6, 7, 3,
                4, 0, 3, 3, 7, 4,
                1, 5, 6, 6, 2, 1
        };

        std::vector<dml::vec3> vertices{
                {-1.0f,  1.0f,  1.0f},
                {-1.0f, -1.0f,  1.0f},
                { 1.0f, -1.0f,  1.0f},
                { 1.0f,  1.0f,  1.0f},
                {-1.0f,  1.0f, -1.0f},
                {-1.0f, -1.0f, -1.0f},
                { 1.0f, -1.0f, -1.0f},
                { 1.0f,  1.0f, -1.0f}
        };

        void resetPipeline() {
            if (pipelineLayout.valid()) pipelineLayout.reset();
            if (pipeline.valid()) pipeline.reset();
        }
    };

    struct DSObject {
        VkhDescriptorPool pool;
        VkhDescriptorSetLayout layout;
        VkhDescriptorSet set;

        std::vector<VkDescriptorSetLayoutBinding> bindings{};
        std::vector<VkDescriptorPoolSize> poolSizes{};

        DSObject() :
            pool(),
            layout(),
            set(pool.v())
        {
        }
    };

    struct DesciptorSetsObj {
        DSObject tlas{};
        DSObject rtTex{};
        DSObject texIndices{};

        DSObject materialTextures{};
        DSObject deferred{};
        DSObject shadowmaps{};
        DSObject camDepth{};
        DSObject camData{};
        DSObject lights{};
        DSObject compTextures{};

        DSObject known{};
    };

    struct PipelineData {
        VkhRenderPass renderPass{};
        VkhPipelineLayout layout{};
        VkhPipeline pipeline{};

        void reset() {
            if (renderPass.valid()) renderPass.reset();
            if (layout.valid()) layout.reset();
            if (pipeline.valid()) pipeline.reset();
        }
    };

    struct SCData {
        VkhSwapchainKHR swapChain{};
        std::vector<VkhImage> images;

        VkFormat imageFormat = VK_FORMAT_UNDEFINED;
        VkViewport viewport{};
        VkExtent2D extent{};

        std::vector<VkhImageView> imageViews;
        std::vector<VkhFramebuffer> framebuffers;

        uint32_t imageCount = 0;
        uint32_t imageIndex = 0;
    };

    struct KeyPO {
        bool pressedLastFrame;
        int keyPress;

        explicit KeyPO(int key) :
            pressedLastFrame(false),
            keyPress(key)
        {
        }

        bool isPressed() {
            bool notPressedLast = !pressedLastFrame;
            pressedLastFrame = (glfwGetKey(window, keyPress) == GLFW_PRESS);
            return (glfwGetKey(window, keyPress) == GLFW_PRESS) && notPressedLast;
        }
    };

    struct WBOITData {
        std::vector<dvl::Texture> weightedColor{};
        std::vector<VkhFramebuffer> frameBuffer{};
    };

    struct LightingData {
        std::vector<dvl::Texture> color;
        std::vector<VkhFramebuffer> frameBuffer;
    };

    struct DeferredRenderingData {
        std::vector<VkhFramebuffer> frameBuffer{};
        std::vector<dvl::Texture> textures{};
        std::vector<dvl::Texture> depth{};
        std::array<VkFormat, 4> colorFormats{};

        size_t colorCount = 0;
    };

    struct CommandBufferCollection {
        std::vector<VkhCommandPool> pools;
        std::vector<VkhCommandBuffer> buffers;
        std::vector<VkCommandBuffer> rawBuffers;

        void reserveClear(size_t size) {
            buffers.clear();
            pools.clear();

            buffers.reserve(size);
            pools.reserve(size);
        }

        size_t size() const {
            return buffers.size();
        }

        VkCommandBuffer* data() {
            if (buffers.size() == 1) {
                return buffers[0].p();
            }
            else {
                rawBuffers.clear();
                rawBuffers.reserve(buffers.size());
                for (const auto& b : buffers) {
                    rawBuffers.push_back(b.v());
                }
                return rawBuffers.data();
            }
        }

        VkhCommandBuffer& operator[](size_t index) {
            return buffers[index];
        }
    };

    struct CommandBufferSet {
        CommandBufferCollection primary{};
        CommandBufferCollection secondary{};
    };

    struct BlasData {
        VkhAccelerationStructure blas{};
        VkhBuffer compBuffer{};
        VkhDeviceMemory compMem{};
    };

    struct TlasData {
        VkhAccelerationStructure as{};
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        VkAccelerationStructureGeometryKHR geometry{};

        VkhBuffer buffer{};
        VkhDeviceMemory mem{};

        VkhBuffer instanceBuffer{};
        VkhDeviceMemory instanceBufferMem{};

        VkhBuffer scratchBuffer{};
        VkhDeviceMemory scratchMem{};
    };

    struct SBT {
        VkhBuffer buffer{};
        VkhDeviceMemory mem{};

        VkDeviceSize size{};
        VkDeviceSize entryS{};

        // sbt regions
        VkStridedDeviceAddressRegionKHR raygenR{};
        VkStridedDeviceAddressRegionKHR missR{};
        VkStridedDeviceAddressRegionKHR hitR{};
        VkStridedDeviceAddressRegionKHR callR{};
    };

    struct TexIndexObj {
        uint32_t albedoIndex = 0;
        uint32_t texBitfield = 0;

        VkDeviceAddress vertAddr = 0;
        VkDeviceAddress indAddr = 0;
    };

    struct TexIndexSSBO {
        TexIndexObj indices[cfg::MAX_MODELS];
    };

    struct FramePushConst {
        int frame;
    };

    struct LightPushConst {
        int lightCount;
        int frameCount;
    };

    struct RTPushConst {
        int frame;
        int lightCount;
    };

    struct ShadowPushConst {
        int frame;
        int lightIndex;
    };

    struct ObjectPushConst {
        int textureExist; // bitfield of which textures exist
        int texIndex; // starting index of the textures in the texture array
    };

private:
    const ShadowMapDim shadowProps{};

    bool rtSupported = false; // a bool if raytracing is supported on the device
    bool rtEnabled = true; // a bool if raytracing has been enabled

    CamData cam{};

    uint32_t maxFrames = 0;
    uint32_t currentFrame = 0;

    // window and rendering context
    VkQueue presentQueue{};
    VkQueue computeQueue{};
    VkQueue transferQueue{};
    vkh::QueueFamilyIndices queueFamilyIndices{};

    // key press objects
    KeyPO escapeKey{ GLFW_KEY_ESCAPE };
    KeyPO eKey{ GLFW_KEY_E };
    KeyPO rKey{ GLFW_KEY_R };
    KeyPO tabKey{ GLFW_KEY_TAB };

    // swap chain
    SCData swap{};

    // pipeline data
    PipelineData deferredPipeline{};
    PipelineData lightingPipeline{};
    PipelineData shadowPipeline{};
    PipelineData compPipeline{};
    PipelineData wboitPipeline{};

    LightingData lightingData{};
    WBOITData wboit{};
    DeferredRenderingData deferredData{};

    // command buffers and command pool
    VkhCommandPool commandPool{};
    CommandBufferSet deferredCommandBuffers{};
    CommandBufferSet lightingPassCommandBuffers{};
    CommandBufferSet shadowMapCommandBuffers{};
    CommandBufferSet wboitCommandBuffers{};
    CommandBufferSet compCommandBuffers{};
    CommandBufferSet rtCommandBuffers{};

    VkhBuffer vertBuffer{};
    VkhBuffer indBuffer{};
    VkhDeviceMemory vertBufferMem{};
    VkhDeviceMemory indBufferMem{};
    VkDeviceSize vertBufferSize = 0;
    VkDeviceSize indBufferSize = 0;

    std::vector<VkhBuffer> lightBuffers{};
    std::vector<VkhDeviceMemory> lightBufferMems{};

    std::vector<VkhBuffer> objInstanceBuffers{};
    std::vector<VkhDeviceMemory> objInstanceBufferMems{};

    VkhBuffer sceneIndexBuffer{};
    VkhDeviceMemory sceneIndexBufferMem{};

    // synchronization primitives
    std::vector<VkhFence> inFlightFences;
    std::vector<VkhSemaphore> imageAvailableSemaphores{};
    std::vector<VkhSemaphore> renderFinishedSemaphores{};
    std::vector<VkhSemaphore> deferredSemaphores{};
    std::vector<VkhSemaphore> shadowSemaphores{};
    std::vector<VkhSemaphore> wboitSemaphores{};
    std::vector<VkhSemaphore> compSemaphores{};
    std::vector<VkhSemaphore> rtSemaphores{};

    FramePushConst framePushConst{};

    // descriptor sets and pools
    DesciptorSetsObj descs{};
    VkhDescriptorPool imguiDescriptorPool{};
    VkhDescriptorSetLayout imguiDescriptorSetLayout{};

    // scene data and objects
    std::vector<vkh::BufData> bufferData;
    std::vector<std::unique_ptr<dvl::Mesh>> objects;
    std::vector<std::unique_ptr<dvl::Mesh>> originalObjects;
    std::vector<uint32_t> playerModels;
    ModelInstanceData objInstanceData{};
    CamUBO camMatData{};
    LightDataSSBO lightData{};
    std::vector<std::unique_ptr<Light>> lights;

    // path tracing
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{};
    std::vector<BlasData> BLAS;
    std::vector<TlasData> tlas{};
    PipelineData rtPipeline{};
    std::vector<dvl::Texture> rtTextures{};
    SBT sbt{};
    TexIndexSSBO texIndices{};
    VkhBuffer texIndicesBuffer{};
    VkhDeviceMemory texIndicesBufferMem{};

    std::vector<VkAccelerationStructureInstanceKHR> meshInstances;

    uint32_t modelIndex = 0; // index of where vertecies are loaded to

    std::unordered_map<size_t, size_t> uniqueModelIndex;
    std::unordered_map<size_t, size_t> modelHashToBufferIndex;

    // textures and materials
    std::vector<dvl::Texture> allTextures;
    std::vector<VkDescriptorImageInfo> shadowInfos;
    std::vector<int> meshTexStartInd;
    uint32_t totalTextureCount = 0;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;

    std::vector<dvl::Texture> compTextures;
    VkSampleCountFlagBits compositionSampleCount = VK_SAMPLE_COUNT_8_BIT;

    // skybox data
    SkyboxObject skybox{};

    // font data
    ImFont* smallFont = nullptr;
    ImFont* largeFont = nullptr;

    // performance metrics
    uint32_t fps = 0;
    double lastFrame = 0.0;

    // multithreading
    std::mutex modelMtx{};

    std::vector<std::future<void>> objTasks;
    std::vector<std::future<void>> textureTasks;
    std::vector<std::future<void>> shadowCmdTasks;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // enable window resizing

        std::string engineName = "3d-vulkan " + cfg::ENGINE_VER;

        window = glfwCreateWindow(cfg::SCREEN_WIDTH, cfg::SCREEN_HEIGHT, engineName.c_str(), nullptr, nullptr);

        // imgui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window, true);

        largeFont = ImGui::GetIO().Fonts->AddFontFromFileTTF((cfg::FONT_DIR + "OpenSans/OpenSans-VariableFont_wdth,wght.ttf").c_str(), 50.0f);
    }

    void validateFiles() {
        // verify that the assets/models/ directory exists. if not, make it
        std::string models = "assets/models/";

        if (!std::filesystem::exists(models)) {
            if (std::filesystem::create_directory(models)) {
#ifdef ENABLE_DEBUG
                std::cout << "Created directory: " << models << "\n";
#endif
            }
            else {
                throw std::runtime_error("Failed to create directory" + models + "!");
            }
        }
    }

    bool isRTSupported(VkPhysicalDevice device) {
        // check if extensions are supported
        bool extensionsSupported = checkExtensionSupport(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            && checkExtensionSupport(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
            && checkExtensionSupport(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

        if (!extensionsSupported) {
            return false;
        }

        // check if the device supports ray tracing pipeline features
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
        rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

        // check if the device supports acceleration structure features
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
        asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        asFeatures.pNext = &rtFeatures;

        VkPhysicalDeviceFeatures2 deviceFeatures2{};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &asFeatures;

        vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);

        return (rtFeatures.rayTracingPipeline == VK_TRUE && asFeatures.accelerationStructure == VK_TRUE);
    }

    void createInstance() {
        VkApplicationInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        instanceInfo.pApplicationName = "My Engine";
        instanceInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        instanceInfo.pEngineName = "No Engine";
        instanceInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        instanceInfo.apiVersion = VK_API_VERSION_1_3;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        const std::vector<const char*> validationLayers = {
            "VK_LAYER_KHRONOS_validation"
        };

        VkInstanceCreateInfo newInfo{};
        newInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        newInfo.pApplicationInfo = &instanceInfo;
        newInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        newInfo.ppEnabledExtensionNames = extensions.data();
        newInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        newInfo.ppEnabledLayerNames = validationLayers.data();

        if (vkCreateInstance(&newInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance!");
        }
    }

    int scoreDevice(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        int score = 0;
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }
        else if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 100;
        }
        return score;
    }

    void pickDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr); // get number of devices
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
        int highestScore = -1;
        for (const auto& device : devices) {
            vkh::QueueFamilyIndices indices = vkh::findQueueFamilyIndices(surface, device);
            if (indices.allComplete()) {
                VkPhysicalDeviceProperties deviceProperties;
                vkGetPhysicalDeviceProperties(device, &deviceProperties);

                int score = scoreDevice(device);
                if (score > highestScore) {
                    bestDevice = device;
                    highestScore = score;
                }
            }
        }
        if (bestDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU for graphics, compute, transfer and presentation!");
        }

        // get all of the queue family indices for the best selected device
        queueFamilyIndices = vkh::findQueueFamilyIndices(surface, bestDevice);

        // use the best device
        physicalDevice = bestDevice;
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

        utils::sep();


        // check if ray tracing is supported
        rtSupported = isRTSupported(physicalDevice);
        rtEnabled = rtEnabled && rtSupported;
        std::cout << "Raytacing is " << (rtEnabled ? "enabled" : "not enabled") << "!\n";
        utils::sep();
    }

    bool checkExtensionSupport(const char* extensionName) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        for (const auto& availableExtension : availableExtensions) {
            if (strcmp(extensionName, availableExtension.extensionName) == 0) {
                return true;
            }
        }
        return false;
    }

    void createLogicalDevice() {
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInf{};
        queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInf.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // index of the graphics queue family
        queueInf.queueCount = 1;
        queueInf.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceNestedCommandBufferFeaturesEXT nestedCommandBufferFeatures{};
        nestedCommandBufferFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT;
        nestedCommandBufferFeatures.nestedCommandBufferSimultaneousUse = VK_TRUE;
        nestedCommandBufferFeatures.nestedCommandBuffer = VK_TRUE;
        nestedCommandBufferFeatures.nestedCommandBufferRendering = VK_TRUE;

        VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
        bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
        bufferDeviceAddressFeatures.pNext = &nestedCommandBufferFeatures;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};

        if (rtEnabled) {
            asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            asFeatures.accelerationStructure = VK_TRUE;
            asFeatures.pNext = &bufferDeviceAddressFeatures;

            rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            rtFeatures.rayTracingPipeline = VK_TRUE;
            rtFeatures.pNext = &asFeatures;

            rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 deviceProperties2{};
            deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            deviceProperties2.pNext = &rtProperties;
            vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.imageCubeArray = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE;
        deviceFeatures.shaderInt64 = VK_TRUE;

        VkPhysicalDeviceDescriptorIndexingFeatures descIndexing{};
        descIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        descIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        descIndexing.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
        descIndexing.runtimeDescriptorArray = VK_TRUE;
        descIndexing.descriptorBindingVariableDescriptorCount = VK_TRUE;
        descIndexing.descriptorBindingPartiallyBound = VK_TRUE;

        if (rtEnabled) descIndexing.pNext = &rtFeatures;
        else descIndexing.pNext = &bufferDeviceAddressFeatures;

        VkDeviceCreateInfo newInfo{};
        newInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        newInfo.pNext = &descIndexing; // add the indexing features to the pNext chain
        newInfo.pQueueCreateInfos = &queueInf;
        newInfo.queueCreateInfoCount = 1;
        newInfo.pEnabledFeatures = &deviceFeatures; //device features to enable

        // specify the device extensions to enable
        std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_KHR_MAINTENANCE3_EXTENSION_NAME,
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME,
            VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        };

        if (rtEnabled) {
            deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        }

        vkhfp::loadFuncPointers(instance);

        for (auto& e : deviceExtensions) {
            if (checkExtensionSupport(e)) {
                std::cout << "---- " << e << " is supported!" << " ----\n";
            }
            else {
                std::cerr << "---- " << e << " is NOT supported!" << " ----\n";
                deviceExtensions.erase(std::remove(deviceExtensions.begin(), deviceExtensions.end(), e), deviceExtensions.end());
            }
        }

        newInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        newInfo.ppEnabledExtensionNames = deviceExtensions.data();
        newInfo.enabledLayerCount = 0;
        newInfo.ppEnabledLayerNames = nullptr;
        VkResult result = vkCreateDevice(physicalDevice, &newInfo, nullptr, &device);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }

        utils::sep();
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void initQueues() {
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.transferFamily.value(), 0, &transferQueue);
    }

    void createSCImageViews() { //create the image views for the swap chain images
        swap.imageViews.resize(swap.imageCount);

        for (size_t i = 0; i < swap.imageCount; i++) {
            if (swap.imageViews[i].valid()) swap.imageViews[i].reset();
            VkImageViewCreateInfo newinfo{};
            newinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            newinfo.image = swap.images[i].v(); // assign the current swap chain image
            newinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            newinfo.format = swap.imageFormat;

            // image will maintain its original component ordering
            newinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            newinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            newinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            newinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            newinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            newinfo.subresourceRange.baseMipLevel = 0;
            newinfo.subresourceRange.levelCount = 1;
            newinfo.subresourceRange.baseArrayLayer = 0;
            newinfo.subresourceRange.layerCount = 1;
            VkResult result = vkCreateImageView(device, &newinfo, nullptr, swap.imageViews[i].p());
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views for the swapchain!!");
            }
        }
    }

    void createSC() {
        vkh::SCsupportDetails swapChainSupport = vkh::querySCsupport(surface);

        // choose the best surface format, present mode, and swap extent for the swap chain
        VkSurfaceFormatKHR surfaceFormat = vkh::chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR present = vkh::chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = vkh::chooseSwapExtent(swapChainSupport.capabilities, cfg::SCREEN_WIDTH, cfg::SCREEN_HEIGHT);

        // get the number of images for the sc. this is the minumum + 1
        swap.imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 && swap.imageCount > swapChainSupport.capabilities.maxImageCount) {
            swap.imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        maxFrames = 3;

        // create the swap chain
        VkSwapchainCreateInfoKHR newinfo{};
        newinfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        newinfo.surface = surface;
        newinfo.minImageCount = swap.imageCount;
        newinfo.imageFormat = surfaceFormat.format;
        newinfo.imageColorSpace = surfaceFormat.colorSpace;
        newinfo.imageExtent = extent;
        newinfo.imageArrayLayers = 1;
        newinfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        // allows only 1 queue family to access the sc at a time
        // this reduces synchronization overhead
        newinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        newinfo.queueFamilyIndexCount = 1;
        newinfo.pQueueFamilyIndices = &queueFamilyIndices.graphicsFamily.value(); // which queue families will handle the swap chain images
        newinfo.preTransform = swapChainSupport.capabilities.currentTransform; // transform to apply to the swap chain before presentation
        newinfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // set the alpha channel to opaque when compositing the final image
        newinfo.presentMode = present;
        newinfo.clipped = VK_TRUE; // if the window is obscured, the pixels that are obscured will not be drawn to
        newinfo.oldSwapchain = VK_NULL_HANDLE;
        if (vkCreateSwapchainKHR(device, &newinfo, nullptr, swap.swapChain.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // get the swap chain images
        vkGetSwapchainImagesKHR(device, swap.swapChain.v(), &swap.imageCount, nullptr);

        swap.images.resize(swap.imageCount);
        std::vector<VkImage> images(swap.imageCount);

        vkGetSwapchainImagesKHR(device, swap.swapChain.v(), &swap.imageCount, images.data());
        for (uint32_t i = 0; i < swap.imageCount; i++) {
            swap.images[i] = VkhImage(images[i]);
            swap.images[i].setDestroy(false); // obj wont be automatically freed when out of scope
        }

        swap.imageFormat = surfaceFormat.format;
        swap.extent = extent;

        createSCImageViews();

        // create the viewport for the swap chain
        swap.viewport.x = 0.0f;
        swap.viewport.y = 0.0f;
        swap.viewport.width = static_cast<float>(swap.extent.width);
        swap.viewport.height = static_cast<float>(swap.extent.height);
        swap.viewport.minDepth = 0.0f;
        swap.viewport.maxDepth = 1.0f;
    }

    void setupFences() {
        inFlightFences.resize(maxFrames);
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // signaled state fence (fence is signaled when created)
        for (size_t i = 0; i < inFlightFences.size(); i++) {
            if (vkCreateFence(device, &fenceInfo, nullptr, inFlightFences[i].p()) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void createSemaphores() {
        for (size_t i = 0; i < maxFrames; i++) {
            imageAvailableSemaphores.push_back(vkh::createSemaphore());
            renderFinishedSemaphores.push_back(vkh::createSemaphore());

            if (!rtEnabled) {
                deferredSemaphores.push_back(vkh::createSemaphore());
                shadowSemaphores.push_back(vkh::createSemaphore());
                wboitSemaphores.push_back(vkh::createSemaphore());
            }
            else {
                rtSemaphores.push_back(vkh::createSemaphore());
            }

            compSemaphores.push_back(vkh::createSemaphore());
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void initializeMouseInput(bool initial) {
        // set the lastX and lastY to the center of the screen
        if (initial) {
            mouse.lastX = static_cast<float>(swap.extent.width) / 2.0f;
            mouse.lastY = static_cast<float>(swap.extent.height) / 2.0f;
            glfwSetCursorPos(window, mouse.lastX, mouse.lastY);
        }

        // only hide and capture cursor if cam.locked is true
        if (mouse.locked) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

            // set the mouse callback
            glfwSetCursorPosCallback(window, mouseCallback);
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }

    static void mouseCallback(GLFWwindow* window, double xPos, double yPos) {
        static bool mouseFirst = true;
        float xp = static_cast<float>(xPos);
        float yp = static_cast<float>(yPos);

        if (mouse.locked) {
            float xoff = mouse.lastX - xp;
            float yoff = mouse.lastY - yp;
            mouse.lastX = xp;
            mouse.lastY = yp;

            float sens = 0.1f;
            xoff *= sens;
            yoff *= sens;

            mouse.rightAngle -= xoff;
            mouse.upAngle -= yoff;
        }
    }

    void handleKeyboardInput() {
        if (mouse.locked) {
            double g = glfwGetTime();
            float deltaTime = static_cast<float>(g - lastFrame);
            lastFrame = g;

            float cameraSpeed = 2.0f * deltaTime;

            mouse.upAngle = fmod(mouse.upAngle + 360.0f, 360.0f);
            mouse.rightAngle = fmod(mouse.rightAngle + 360.0f, 360.0f);

            cam.updateQuaternion(mouse);
            dml::vec3 forward = dml::quatToDir(cam.quat);
            dml::vec3 right = dml::normalize(dml::cross(forward, dml::vec3(0, 1, 0)));

            // camera movement
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                cam.pos -= forward * cameraSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                cam.pos += forward * cameraSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                cam.pos += right * cameraSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                cam.pos -= right * cameraSpeed;
            }

            // up and down movement
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                cam.pos.y -= 1 * cameraSpeed;
            }
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
                cam.pos.y += 1 * cameraSpeed;
            }
        }

        // realtime object loading
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            summonModel();
        }

        // realtime light loading
        if (eKey.isPressed()) {
            summonLight();
        }

        if (rKey.isPressed()) {
            size_t vertCount = 0;
            for (const auto& o : objects) {
                vertCount += o->vertices.size();
            }

            double score = fps * (((vertCount) / 50000.0) + std::pow(lights.size(), 1.3) + (objects.size() / 10.0));

            std::cout << "Vertex count: " << vertCount << "\n";
            std::cout << "Object count: " << objects.size() << "\n";
            std::cout << "Light count: " << lights.size() << " / " << cfg::MAX_LIGHTS << "\n";
            std::cout << "Score: " << score << "\n";
            utils::sep();
        }

        if (tabKey.isPressed()) {
            resetScene();
        }

        // lock / unlock mouse
        if (escapeKey.isPressed()) {
            mouse.locked = !mouse.locked;
            initializeMouseInput(mouse.locked);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void getGLTFImageData(dvl::Texture& tex) {
        int width = tex.gltfImage->width;
        int height = tex.gltfImage->height;
        int channels = tex.gltfImage->component;

        std::shared_ptr<unsigned char[]> data = std::shared_ptr<unsigned char[]>(new unsigned char[width * height * 4]);

        // only images with 3 or 4 channels are supported
        if (channels != 4 && channels != 3) {
            throw std::runtime_error("unsupported number of channels!");
        }

        // iterate through the image data and copy it into the new array
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    // copy the data from the original image into the new array
                    data[(y * width + x) * 4 + c] = tex.gltfImage->image[(y * width + x) * channels + c];
                }
                // if the original image doesn't have an alpha channel, set alpha to 255 (completely opaque)
                if (channels == 3) {
                    data[(y * width + x) * 4 + 3] = 255;
                }
            }
        }
        tex.width = width;
        tex.height = height;

        tex.gltfImage.reset();
        tex.rawData = data;
    }

    void createImageStagingBuffer(dvl::Texture& tex, bool cubeMap) {
        auto bpp = cubeMap ? sizeof(float) * 4 : 4;
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(tex.width) * tex.height * bpp;

        if (cubeMap) {
            vkh::createAndWriteHostBuffer(tex.stagingBuffer, tex.stagingBufferMem, skybox.imgData, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
        else {
            vkh::createAndWriteHostBuffer(tex.stagingBuffer, tex.stagingBufferMem, tex.rawData.get(), imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
    }

    void createTexturedImage(dvl::Texture& tex, bool doMipmap, vkh::TextureType type = vkh::BASE) {
        if (tex.stagingBuffer.valid()) return;
        createImageStagingBuffer(tex, false);

        tex.mipLevels = doMipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(tex.width, tex.height)))) + 1 : 1;
        VkFormat imgFormat;
        switch (type) {
        case vkh::BASE:
        case vkh::EMISSIVE:
            imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
            break;
        default:
            imgFormat = VK_FORMAT_R8G8B8A8_UNORM;
            break;
        }

        vkh::createImage(tex.image, tex.memory, tex.width, tex.height, imgFormat, tex.mipLevels, 1, false, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, tex.sampleCount);

        // init the VkBufferImageCopy struct for the styaging buffer to image copying
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //specifies the aspect of the image to copy
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { static_cast<uint32_t>(tex.width), static_cast<uint32_t>(tex.height), 1 };

        VkhCommandBuffer tempBuffer = vkh::beginSingleTimeCommands(commandPool);

        vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, tex.mipLevels, 0);
        vkCmdCopyBufferToImage(tempBuffer.v(), tex.stagingBuffer.v(), tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); //copy the data from the staging buffer to the image

        int mipWidth = tex.width;
        int mipHeight = tex.height;

        // create mipmaps for the image if enabled
        if (doMipmap) {
            for (uint32_t j = 0; j < tex.mipLevels; j++) {
                vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1, j);

                // if the cutrrent mip level isnt the last, blit the image to generate the next mip level
                // bliting is the process of transfering the image data from one image to another usually with a form of scaling or filtering
                if (j < tex.mipLevels - 1) {
                    VkImageBlit blit{};
                    blit.srcOffsets[0] = { 0, 0, 0 };
                    blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
                    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    blit.srcSubresource.mipLevel = j;
                    blit.srcSubresource.baseArrayLayer = 0;
                    blit.srcSubresource.layerCount = 1;
                    blit.dstOffsets[0] = { 0, 0, 0 };

                    // divide the width and height by 2 if over 1
                    blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };

                    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    blit.dstSubresource.mipLevel = j + 1;
                    blit.dstSubresource.baseArrayLayer = 0;
                    blit.dstSubresource.layerCount = 1;
                    vkCmdBlitImage(tempBuffer.v(), tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
                }

                vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, j);

                //for the next mip level, divide the width and height by 2, unless they are already 1
                if (mipWidth > 1) mipWidth /= 2;
                if (mipHeight > 1) mipHeight /= 2;
            }
        }
        else {
            vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, tex.mipLevels, 0);
        }

        vkh::endSingleTimeCommands(tempBuffer, commandPool, graphicsQueue);

        tex.rawData.reset();
    }

    void getImageDataHDR(const std::string& path, dvl::Texture& t, float*& imgData) {
        int texWidth, texHeight, texChannels;
        imgData = stbi_loadf(path.c_str(), &texWidth, &texHeight, &texChannels, 4);
        t.width = texWidth;
        t.height = texHeight;
        if (!imgData) {
            std::string error = stbi_failure_reason();
            throw std::runtime_error("failed to load HDR image: " + path + "! Reason: " + error);
        }
    }

    void freeHDRImage(float*& imgData) {
        stbi_image_free(imgData);
        imgData = nullptr;
    }

    void createTexturedCubemap(dvl::Texture& tex, float*& imgData) {
        getImageDataHDR(tex.path, tex, imgData);
        if (imgData == nullptr) {
            throw std::runtime_error("failed to load image data!");
        }
        createImageStagingBuffer(tex, true);

        // calculate the size of one face of the cubemap
        uint32_t faceWidth = tex.width / 4;
        uint32_t faceHeight = tex.height / 3;
        auto bpp = sizeof(float) * 4; // four floats for R32G32B32A32_SFLOAT
        VkDeviceSize faceSize = static_cast<VkDeviceSize>(faceWidth) * faceHeight * bpp;

        // ensure the atlas dimensions are valid for a horizontal cross layout
        if (faceHeight != faceWidth) {
            throw std::runtime_error("Cubemap atlas dimensions are invalid!!");
        }

        vkh::createImage(tex.image, tex.memory, faceWidth, faceHeight, VK_FORMAT_R32G32B32A32_SFLOAT, 1, 6, true, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, tex.sampleCount);

        vkh::transitionImageLayout(commandPool, tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, 1, 0);
        VkhCommandBuffer copyCmdBuffer = vkh::beginSingleTimeCommands(commandPool);

        std::array<VkBufferImageCopy, 6> regions;
        std::array<std::pair<uint32_t, uint32_t>, 6> faceOffsets = { {{2, 1}, {0, 1}, {1, 0}, {1, 2}, {1, 1}, {3, 1}} };

        for (uint32_t i = 0; i < regions.size(); i++) {
            VkBufferImageCopy& region = regions[i];

            uint32_t offsetX = faceOffsets[i].first * faceWidth;
            uint32_t offsetY = faceOffsets[i].second * faceHeight;

            region.bufferOffset = offsetY * tex.width * bpp + offsetX * bpp;
            region.bufferRowLength = tex.width;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = i;
            region.imageSubresource.layerCount = 1;

            region.imageOffset = { 0, 0, 0 };
            region.imageExtent = { faceWidth, faceHeight, 1 };

            vkCmdCopyBufferToImage(copyCmdBuffer.v(), tex.stagingBuffer.v(), tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        vkh::endSingleTimeCommands(copyCmdBuffer, commandPool, graphicsQueue);

        vkh::transitionImageLayout(commandPool, tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6, 1, 0);
        freeHDRImage(imgData);
    }

    void createMeshTexture(dvl::Texture& tex, vkh::TextureType type) {
        if (tex.found) {
            createTexturedImage(tex, true, type);
            vkh::createImageView(tex, type);
            vkh::createSampler(tex.sampler, tex.mipLevels);
        }
    }

    void createMeshTextures(std::unique_ptr<dvl::Mesh>& newObject) {
        createMeshTexture(newObject->material.baseColor, vkh::BASE);
        createMeshTexture(newObject->material.metallicRoughness, vkh::METALLIC);
        createMeshTexture(newObject->material.normalMap, vkh::NORMAL);
        createMeshTexture(newObject->material.emissiveMap, vkh::EMISSIVE);
        createMeshTexture(newObject->material.occlusionMap, vkh::OCCLUSION);
    }

    void loadTexImageData(dvl::Texture& t) {
        if (t.found) {
            textureTasks.emplace_back(std::async(std::launch::async, &Engine::getGLTFImageData, this, std::ref(t)));
        }
    }

    void createTexture(dvl::Texture& tex, VkFormat format, VkImageUsageFlags usage, uint32_t width, uint32_t height) {
        vkh::createImage(tex.image, tex.memory, width, height, format, 1, 1, false, usage, tex.sampleCount);
        vkh::createImageView(tex, format);
        vkh::createSampler(tex.sampler, tex.mipLevels);
    }

    void createTexture(dvl::Texture& tex, vkh::TextureType type, VkImageUsageFlags usage, uint32_t width, uint32_t height) {
        vkh::createImage(tex.image, tex.memory, width, height, type, 1, 1, false, usage, tex.sampleCount);
        vkh::createImageView(tex, type);
        vkh::createSampler(tex.sampler, tex.mipLevels, type);
    }

    void setupTextures(bool shadow) {
        depthFormat = vkh::findDepthFormat();

        // composition images
        for (size_t i = 0; i < swap.imageCount; i++) {
            compTextures.emplace_back(compositionSampleCount);
            createTexture(compTextures[i], swap.imageFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, swap.extent.width, swap.extent.height);
        }

        if (rtEnabled) {
            // rt image
            rtTextures.resize(maxFrames);
            for (size_t i = 0; i < maxFrames; i++) {
                createTexture(rtTextures[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, swap.extent.width, swap.extent.height);
                vkh::transitionImageLayout(commandPool, rtTextures[i].image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1, 1, 0);
            }
        }
        else {
            lightingData.color.resize(maxFrames);
            wboit.weightedColor.resize(maxFrames);

            // deferred rendering textures
            deferredData.colorCount = maxFrames * 4;
            deferredData.depth.resize(maxFrames);
            deferredData.textures.resize(deferredData.colorCount);
            VkImageUsageFlags deferredUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

            for (size_t i = 0; i < maxFrames; i++) {

                // lighting textures
                createTexture(lightingData.color[i], swap.imageFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, swap.extent.width, swap.extent.height);

                // wboit
                createTexture(wboit.weightedColor[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, swap.extent.width, swap.extent.height);

                // shadowmaps
                if (shadow) {
                    for (size_t j = 0; j < lights.size(); j++) {
                        dvl::Texture shadowMap{};
                        createTexture(shadowMap, vkh::DEPTH, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowProps.width, shadowProps.height);
                        lights[j]->shadowMapData.push_back(shadowMap);
                    }
                }

                createTexture(deferredData.depth[i], vkh::DEPTH, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, swap.extent.width, swap.extent.height);

                for (size_t j = 0; j < 4; j++) {
                    size_t texIndex = (i * 4) + j;
                    deferredData.colorFormats[j] = vkh::getTextureFormat(vkh::TextureType(j));
                    createTexture(deferredData.textures[texIndex], deferredData.colorFormats[j], deferredUsage, swap.extent.width, swap.extent.height);
                }
            }
        }
    }

    void loadSkybox(const std::string& path) {
        skybox.cubemap.path = cfg::SKYBOX_DIR + path;
        createTexturedCubemap(skybox.cubemap, skybox.imgData);

        vkh::createImageView(skybox.cubemap, vkh::CUBEMAP);
        vkh::createSampler(skybox.cubemap.sampler, skybox.cubemap.mipLevels, vkh::CUBEMAP);

        skybox.bufferData.vertexOffset = 0;
        skybox.bufferData.vertexCount = 8;
        skybox.bufferData.indexOffset = 0;
        skybox.bufferData.indexCount = 36;
    }

    void getTexIndices() {
        uint32_t p = 0;
        for (size_t i = 0; i < objects.size(); i++) {
            if (uniqueModelIndex[objects[i]->meshHash] == i) {
                TexIndexObj& obj = texIndices.indices[i];

                uint32_t bitfield = 0;
                bitfield |= (objects[i]->material.baseColor.found ? 1 : 0);
                bitfield |= (objects[i]->material.metallicRoughness.found ? 1 : 0) << 1;
                bitfield |= (objects[i]->material.normalMap.found ? 1 : 0) << 2;
                bitfield |= (objects[i]->material.emissiveMap.found ? 1 : 0) << 3;
                bitfield |= (objects[i]->material.occlusionMap.found ? 1 : 0) << 4;

                obj.texBitfield = bitfield;
                obj.albedoIndex = meshTexStartInd[p];

                size_t bufferInd = modelHashToBufferIndex[objects[i]->meshHash];

                obj.vertAddr = vkh::bufferDeviceAddress(vertBuffer) + (bufferData[bufferInd].vertexOffset * sizeof(dvl::Vertex));
                obj.indAddr = vkh::bufferDeviceAddress(indBuffer) + (bufferData[bufferInd].indexOffset * sizeof(uint32_t));
                p++;
            }
        }

        VkhBuffer stagingBuffer;
        VkhDeviceMemory stagingBufferMem;
        vkh::createAndWriteHostBuffer(stagingBuffer, stagingBufferMem, &texIndices, sizeof(TexIndexSSBO), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        vkh::copyBuffer(stagingBuffer, texIndicesBuffer, commandPool, graphicsQueue, sizeof(TexIndexSSBO));
    }

    void getAllTextures() {
        size_t currentIndex = 0;
        for (size_t i = 0; i < objects.size(); i++) {
            auto& obj = objects[i];
            if (uniqueModelIndex[obj->meshHash] == i) {
                meshTexStartInd.push_back(static_cast<int>(currentIndex));
                if (obj->material.baseColor.found) {
                    allTextures.push_back(obj->material.baseColor);
                    currentIndex++;
                }
                if (obj->material.metallicRoughness.found) {
                    allTextures.push_back(obj->material.metallicRoughness);
                    currentIndex++;
                }
                if (obj->material.normalMap.found) {
                    allTextures.push_back(obj->material.normalMap);
                    currentIndex++;
                }
                if (obj->material.emissiveMap.found) {
                    allTextures.push_back(obj->material.emissiveMap);
                    currentIndex++;
                }
                if (obj->material.occlusionMap.found) {
                    allTextures.push_back(obj->material.occlusionMap);
                    currentIndex++;
                }
            }
        }

        totalTextureCount = static_cast<uint32_t>(allTextures.size());

        if (rtEnabled) {
            getTexIndices();
        }

        std::cout << "Finished loading " << totalTextureCount << " textures!\n";
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void loadModel(const std::string& path, dml::vec3 scale, dml::vec4 rot, dml::vec3 pos) {
        uint32_t meshInd = 0; // index of the mesh in the model

        tinygltf::Model gltfModel;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
        LOG_WARNING_IF(warn, !warn.empty());

        if (!err.empty()) {
            std::cerr << err << "\n";
            return;
        }
        if (!ret) {
            std::cerr << "Failed to load GLTF model\n";
            return;
        }

        // get the index of the parent node for each node
        std::unordered_map<int, int> parentInd;
        for (size_t nodeIndex = 0; nodeIndex < gltfModel.nodes.size(); nodeIndex++) {
            const auto& node = gltfModel.nodes[nodeIndex];
            for (const auto& childIndex : node.children) {
                parentInd[childIndex] = static_cast<int>(nodeIndex);
            }
        }

        // check if the model has any skins or animations (not supported for now)
        LOG_WARNING_IF("The " + path + " contains skinning information", !gltfModel.skins.empty());
        LOG_WARNING_IF("The " + path + " contains animation data", !gltfModel.animations.empty());
        LOG_WARNING_IF("The " + path + " contains cameras", !gltfModel.cameras.empty());

        // check if the gltf model relies on any extensions
        for (const auto& extension : gltfModel.extensionsUsed) {
            LOG_WARNING("The" + path + " relies on: " + extension);
        }

        objects.reserve(gltfModel.meshes.size());
        for (const auto& mesh : gltfModel.meshes) {
            dvl::Mesh m = dvl::loadMesh(mesh, gltfModel, parentInd, meshInd++, scale, pos, rot);
            modelMtx.lock();
            objects.push_back(std::make_unique<dvl::Mesh>(m));
            modelMtx.unlock();
            modelIndex++;
        }
    }

    void createObject(const std::string& name, dml::vec3 scale, dml::vec4 rot, dml::vec3 pos) {
        std::string path = std::string(cfg::MODEL_DIR) + name;
        objTasks.emplace_back(std::async(std::launch::async, &Engine::loadModel, this, path, scale, rot, pos));
    }

    Light createLight(dml::vec3 pos, dml::vec3 t, dml::vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 2.0f) {
        Light l;
        l.data.col = color;
        l.data.pos = pos;
        l.data.intensity = intensity;
        l.data.target = t;

        return l;
    }

    Light createPlayerLight(dml::vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 2.0f) {
        Light l;
        l.data.col = color;
        l.data.intensity = intensity;
        l.followPlayer = true;

        return l;
    }

    void setPlayer(uint16_t i) {
        auto p = std::make_unique<dvl::Mesh>(*objects[i]);
        p->player = true;
        p->scale = dml::vec3(0.3f, 0.3f, 0.3f);
        p->position = dml::vec3(-3.0f, 0.0f, 3.0f);
        playerModels.push_back(i);
        objects.push_back(std::move(p));
    }

    void loadScene() {
        objTasks.clear();
        textureTasks.clear();

        // load each mesh
        createObject("boat.glb", { 18.4f, 18.4f, 18.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.23f, 0.0f, 3.11f });
        createObject("knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f });
        /*createObject("sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQuat({ 3.0f, 1.0f, -2.11f }, { 0.0f, 0.0f, 0.0f }), { 3.0f, 1.0f, -2.11f });
        createObject("sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQuat({ -2.0f, 0.0f, 2.11f }, { 0.0f, 0.0f, 0.0f }), { -2.0f, 0.0f, 2.11f });*/

        for (auto& t : objTasks) {
            t.wait();
        }

        if (!objects.size()) {
            throw std::runtime_error("Failed to load models!");
        }

        utils::sep();

        textureTasks.reserve(objects.size() * 5);

        // load the raw image data of each image
        for (const auto& m : objects) {
            loadTexImageData(m->material.baseColor);
            loadTexImageData(m->material.metallicRoughness);
            loadTexImageData(m->material.normalMap);
            loadTexImageData(m->material.emissiveMap);
            loadTexImageData(m->material.occlusionMap);
        }

        for (auto& t : textureTasks) {
            t.wait();
        }

        // create textires
        for (auto& m : objects) {
            createMeshTextures(m);
        }

        // create all lights
        lights.push_back(std::make_unique<Light>(createPlayerLight()));

        originalObjects.reserve(objects.size());
        for (auto& obj : objects) {
            originalObjects.push_back(std::make_unique<dvl::Mesh>(*obj));
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    void setupBuffers() {
        lightBuffers.resize(maxFrames);
        lightBufferMems.resize(maxFrames);

        objInstanceBuffers.resize(maxFrames);
        objInstanceBufferMems.resize(maxFrames);

        cam.buffers.resize(maxFrames);
        cam.bufferMems.resize(maxFrames);

        for (size_t i = 0; i < maxFrames; i++) {
            vkh::createHostVisibleBuffer(lightBuffers[i], lightBufferMems[i], sizeof(LightDataSSBO), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            vkh::createHostVisibleBuffer(objInstanceBuffers[i], objInstanceBufferMems[i], sizeof(ModelInstanceData), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            vkh::createHostVisibleBuffer(cam.buffers[i], cam.bufferMems[i], sizeof(CamUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }

        // skybox buffer data
        vkh::createAndWriteLocalBuffer(skybox.vertBuffer, skybox.vertBufferMem, skybox.vertices.data(), sizeof(dml::vec3) * skybox.bufferData.vertexCount, commandPool, graphicsQueue, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        vkh::createAndWriteLocalBuffer(skybox.indBuffer, skybox.indBufferMem, skybox.indices.data(), sizeof(uint32_t) * skybox.bufferData.indexCount, commandPool, graphicsQueue, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        if (rtEnabled) {
            vkh::createDeviceLocalBuffer(texIndicesBuffer, texIndicesBufferMem, sizeof(TexIndexSSBO), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
    }

    void calcCameraMats() {
        cam.viewMatrix = cam.getViewMatrix(mouse);
        cam.projectionMatrix = dml::projection(cam.fov, swap.extent.width / static_cast<float>(swap.extent.height), cam.nearP, cam.farP);
    }

    void calcShadowMats(Light& l) {
        if (l.followPlayer) {
            l.data.pos = dml::getCamWorldPos(cam.viewMatrix);
            l.data.target = l.data.pos + dml::quatToDir(cam.quat);
        }
        // spotlight shadow mapping math code
        float aspectRatio = static_cast<float>(shadowProps.width) / static_cast<float>(shadowProps.height);
        float nearPlane = 0.01f, farPlane = 100.0f;

        dml::vec3 up = dml::vec3(0.0f, 1.0f, 0.0f);
        if (l.data.pos == l.data.target) {
            std::cerr << "Light position and target are the same!\n";
            return;
        }

        l.data.view = dml::lookAt(l.data.pos, l.data.target, up);

        float fov = dml::degrees(l.data.outerConeAngle) * 2.0f;
        l.data.proj = dml::projection(fov, aspectRatio, nearPlane, farPlane);
    }

    void updateUBO() {
        // calc matricies for lights
        for (size_t i = 0; i < lights.size(); i++) {
            Light& l = *lights[i];
            calcShadowMats(l);
            lightData.lightCords[i] = l.data;
        }

        vkh::writeBuffer(lightBufferMems[currentFrame], &lightData.lightCords, sizeof(LightDataObject) * lights.size());

        // calc matricies for camera
        calcCameraMats();
        dml::mat4 view = cam.viewMatrix;
        dml::mat4 proj = cam.projectionMatrix;
        if (rtEnabled) {
            view = dml::inverseMatrix(cam.viewMatrix);
            proj = dml::inverseMatrix(cam.projectionMatrix);
        }

        camMatData.view = view;
        camMatData.proj = proj;

        vkh::writeBuffer(cam.bufferMems[currentFrame], &camMatData, sizeof(camMatData));

        // calc matricies for objects
        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i]->player) {
                dml::mat4 t = dml::translate(cam.pos);
                dml::mat4 r;
                dml::mat4 s = dml::scale(objects[i]->scale);
                dml::mat4 model = (t * r * s) * objects[i]->modelMatrix;

                objInstanceData.object[i].model = model;
                objInstanceData.object[i].render = 1;
            }
            else {
                objInstanceData.object[i].model = objects[i]->modelMatrix;
                objInstanceData.object[i].render = 0;
            }
        }

        vkh::writeBuffer(objInstanceBufferMems[currentFrame], &objInstanceData, sizeof(objInstanceData));
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void initDescriptorSet(DSObject& obj, bool variableDescriptorCount) {
        if (obj.set.valid()) obj.set.reset(obj.pool.v());

        vkh::createDSLayout(obj.layout, obj.bindings.data(), obj.bindings.size(), variableDescriptorCount, false);
        vkh::createDSPool(obj.pool, obj.poolSizes.data(), obj.poolSizes.size());

        uint32_t size = 0;
        if (variableDescriptorCount) {
            size = obj.bindings.back().descriptorCount;
        }

        obj.set = vkh::allocDS(obj.layout, obj.pool, size);
    }

    void initDescriptorInfo(DSObject& obj, VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding, size_t descriptorCount) {
        obj.bindings.push_back(vkh::createDSLayoutBinding(binding, descriptorCount, type, stageFlags));
        obj.poolSizes.push_back(vkh::createDSPoolSize(descriptorCount, type));
    }

    void initDSInfo() {
        VkShaderStageFlags textursSS{};
        VkShaderStageFlags lightDataSS{};
        VkShaderStageFlags skyboxSS{};
        VkShaderStageFlags camSS{};

        if (rtEnabled) {
            textursSS = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            lightDataSS = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            skyboxSS = VK_SHADER_STAGE_MISS_BIT_KHR;
            camSS = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        }
        else {
            textursSS = VK_SHADER_STAGE_FRAGMENT_BIT;
            lightDataSS = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            skyboxSS = VK_SHADER_STAGE_FRAGMENT_BIT;
            camSS = VK_SHADER_STAGE_VERTEX_BIT;
        }

        initDescriptorInfo(descs.rtTex, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT, 0, maxFrames);
        initDescriptorInfo(descs.tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, maxFrames);
        initDescriptorInfo(descs.texIndices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, totalTextureCount);

        initDescriptorInfo(descs.materialTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textursSS, 0, totalTextureCount);
        initDescriptorInfo(descs.camData, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, camSS, 0, maxFrames);
        initDescriptorInfo(descs.lights, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, lightDataSS, 0, maxFrames);
        initDescriptorInfo(descs.deferred, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, deferredData.colorCount);
        initDescriptorInfo(descs.shadowmaps, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, cfg::MAX_LIGHTS * maxFrames);
        initDescriptorInfo(descs.camDepth, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, maxFrames);
        initDescriptorInfo(descs.compTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, maxFrames * 2);

        initDescriptorInfo(descs.known, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, skyboxSS, 0, 1);
    }

    void updateDescriptorSets() {
        uint32_t lightSize = static_cast<uint32_t>(lights.size());

        std::vector<VkDescriptorImageInfo> imageInfos;
        imageInfos.reserve(totalTextureCount);

        for (size_t i = 0; i < totalTextureCount; i++) {
            imageInfos.push_back(vkh::createDSImageInfo(allTextures[i].imageView, allTextures[i].sampler));
        }

        std::vector<VkDescriptorBufferInfo> lightBufferInfos{};
        std::vector<VkDescriptorBufferInfo> camBufferInfos{};

        lightBufferInfos.reserve(maxFrames);
        camBufferInfos.reserve(maxFrames);

        for (size_t i = 0; i < maxFrames; i++) {
            VkDescriptorBufferInfo linfo{};
            linfo.buffer = lightBuffers[i].v();
            linfo.offset = 0;
            linfo.range = sizeof(LightDataSSBO);
            lightBufferInfos.push_back(linfo);

            VkDescriptorBufferInfo cinfo{};
            cinfo.buffer = cam.buffers[i].v();
            cinfo.offset = 0;
            cinfo.range = sizeof(CamUBO);
            camBufferInfos.push_back(cinfo);
        }

        VkDescriptorImageInfo skyboxInfo = vkh::createDSImageInfo(skybox.cubemap.imageView, skybox.cubemap.sampler);

        // rasterization
        std::vector<VkDescriptorImageInfo> compositionPassImageInfo{};
        std::vector<VkDescriptorImageInfo> deferredImageInfo{};
        std::vector<VkDescriptorImageInfo> depthInfo{};

        // raytracing
        VkWriteDescriptorSetAccelerationStructureKHR tlasInfo{};
        std::vector<VkAccelerationStructureKHR> tlasList{};

        std::vector<VkDescriptorImageInfo> rtPresentTextures{};
        VkDescriptorBufferInfo texIndexInfo{};

        if (rtEnabled) {
            tlasList.reserve(maxFrames);
            rtPresentTextures.reserve(maxFrames);

            for (size_t i = 0; i < maxFrames; i++) {
                tlasList.push_back(tlas[i].as.v());
                rtPresentTextures.push_back(vkh::createDSImageInfo(rtTextures[i].imageView, rtTextures[i].sampler, VK_IMAGE_LAYOUT_GENERAL));
            }

            tlasInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            tlasInfo.pAccelerationStructures = tlasList.data();
            tlasInfo.accelerationStructureCount = maxFrames;

            texIndexInfo.buffer = texIndicesBuffer.v();
            texIndexInfo.offset = 0;
            texIndexInfo.range = sizeof(TexIndexSSBO);
        }
        else {
            compositionPassImageInfo.reserve(maxFrames * 2);
            deferredImageInfo.reserve(deferredData.colorCount);
            shadowInfos.reserve(maxFrames * lightSize);
            depthInfo.reserve(maxFrames);

            for (size_t i = 0; i < lights.size(); i++) {
                for (size_t j = 0; j < maxFrames; j++) {
                    dvl::Texture& tex = lights[i]->shadowMapData[j];
                    shadowInfos.push_back(vkh::createDSImageInfo(tex.imageView, tex.sampler));
                }
            }

            for (size_t i = 0; i < maxFrames; i++) {
                for (size_t j = 0; j < 4; j++) {
                    size_t k = (i * 4) + j;

                    dvl::Texture& tex = deferredData.textures[k];
                    deferredImageInfo.push_back(vkh::createDSImageInfo(tex.imageView, tex.sampler));
                }

                depthInfo.push_back(vkh::createDSImageInfo(deferredData.depth[i].imageView, deferredData.depth[i].sampler));
                compositionPassImageInfo.push_back(vkh::createDSImageInfo(lightingData.color[i].imageView, lightingData.color[i].sampler));
                compositionPassImageInfo.push_back(vkh::createDSImageInfo(wboit.weightedColor[i].imageView, wboit.weightedColor[i].sampler));
            }
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites{};
        if (rtEnabled) {
            descriptorWrites.push_back(vkh::createDSWrite(descs.rtTex.set, 0, descs.rtTex.bindings[0].descriptorType, rtPresentTextures.data(), rtPresentTextures.size()));
            descriptorWrites.push_back(vkh::createDSWrite(descs.tlas.set, 0, descs.tlas.bindings[0].descriptorType, &tlasInfo, maxFrames));
            descriptorWrites.push_back(vkh::createDSWrite(descs.texIndices.set, 0, descs.texIndices.bindings[0].descriptorType, texIndexInfo));
        }
        else {
            descriptorWrites.push_back(vkh::createDSWrite(descs.deferred.set, 0, descs.deferred.bindings[0].descriptorType, deferredImageInfo.data(), deferredImageInfo.size()));
            descriptorWrites.push_back(vkh::createDSWrite(descs.shadowmaps.set, 0, descs.shadowmaps.bindings[0].descriptorType, shadowInfos.data(), shadowInfos.size()));
            descriptorWrites.push_back(vkh::createDSWrite(descs.camDepth.set, 0, descs.camDepth.bindings[0].descriptorType, depthInfo.data(), depthInfo.size()));
            descriptorWrites.push_back(vkh::createDSWrite(descs.compTextures.set, 0, descs.compTextures.bindings[0].descriptorType, compositionPassImageInfo.data(), compositionPassImageInfo.size()));
        }

        descriptorWrites.push_back(vkh::createDSWrite(descs.materialTextures.set, 0, descs.materialTextures.bindings[0].descriptorType, imageInfos.data(), imageInfos.size()));
        descriptorWrites.push_back(vkh::createDSWrite(descs.camData.set, 0, descs.camData.bindings[0].descriptorType, camBufferInfos.data(), camBufferInfos.size()));
        descriptorWrites.push_back(vkh::createDSWrite(descs.lights.set, 0, descs.lights.bindings[0].descriptorType, lightBufferInfos.data(), lightBufferInfos.size()));
        descriptorWrites.push_back(vkh::createDSWrite(descs.known.set, 0, descs.known.bindings[0].descriptorType, skyboxInfo));

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    void setupDescriptorSets() {
        initDSInfo();

        if (rtEnabled) {
            initDescriptorSet(descs.rtTex, true);
            initDescriptorSet(descs.tlas, true);
            initDescriptorSet(descs.texIndices, true);
        }
        else {
            initDescriptorSet(descs.deferred, true);
            initDescriptorSet(descs.shadowmaps, true);
            initDescriptorSet(descs.camDepth, true);
            initDescriptorSet(descs.compTextures, true);
        }

        initDescriptorSet(descs.materialTextures, true);
        initDescriptorSet(descs.camData, true);
        initDescriptorSet(descs.lights, true);
        initDescriptorSet(descs.known, false);

        updateDescriptorSets();
    }

    void updateLightDS() {
        VkWriteDescriptorSet dw = vkh::createDSWrite(descs.shadowmaps.set, 0, descs.shadowmaps.bindings[0].descriptorType, shadowInfos.data(), shadowInfos.size());
        vkUpdateDescriptorSets(device, 1, &dw, 0, nullptr);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<char> readFile(const std::string& filename) { //outputs binary data from a SPIRV shader file
        std::ifstream file(filename, std::ios::ate | std::ios::binary); //ate means start reading at the end of the file and binary means read the file as binary
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        size_t fileSize = static_cast<size_t>(file.tellg()); //tellg gets the position of the read/write head
        std::vector<char> buffer(fileSize);
        file.seekg(0); //seekg sets the position of the read/write head
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    VkhShaderModule createShaderMod(const std::string& name) {
        std::vector<char> shaderCode = readFile(cfg::SHADER_DIR + name + std::string("_shader.spv"));
        return vkh::createShaderModule(shaderCode);
    }


    void createDeferredPipeline() {
        VkhShaderModule vertShaderModule = createShaderMod("deferred_vertex");
        VkhShaderModule fragShaderModule = createShaderMod("deferred_fragment");

        VkPipelineShaderStageCreateInfo vertStage = vkh::createShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
        VkPipelineShaderStageCreateInfo fragStage = vkh::createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertStage, fragStage };

        VkVertexInputBindingDescription vertBindDesc{};
        vertBindDesc.binding = 0;
        vertBindDesc.stride = sizeof(dvl::Vertex);
        vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputBindingDescription instanceBindDesc{};
        instanceBindDesc.binding = 1;
        instanceBindDesc.stride = sizeof(ModelInstance);
        instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };

        std::array<VkVertexInputAttributeDescription, 9> attrDesc{};
        attrDesc[0].binding = 0;
        attrDesc[0].location = 0;
        attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
        attrDesc[0].offset = offsetof(dvl::Vertex, pos);

        // texture coordinates
        attrDesc[1].binding = 0;
        attrDesc[1].location = 1;
        attrDesc[1].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
        attrDesc[1].offset = offsetof(dvl::Vertex, tex);

        // normal
        attrDesc[2].binding = 0;
        attrDesc[2].location = 2;
        attrDesc[2].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for normal
        attrDesc[2].offset = offsetof(dvl::Vertex, normal);

        // tangents
        attrDesc[3].binding = 0;
        attrDesc[3].location = 3;
        attrDesc[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for tangent
        attrDesc[3].offset = offsetof(dvl::Vertex, tangent);

        // pass the model matrix as a per-instance data
        // seperate the matrix into 4 vec4's so it can be quickly passed and processed
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t index = 4 + i;
            attrDesc[index].binding = 1;
            attrDesc[index].location = index;
            attrDesc[index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attrDesc[index].offset = offsetof(ModelInstance, model) + sizeof(float) * 4 * i;
        }

        // render flag
        attrDesc[8].binding = 1;
        attrDesc[8].location = 8;
        attrDesc[8].format = VK_FORMAT_R32_UINT; // 1 uint32_t
        attrDesc[8].offset = offsetof(ModelInstance, render);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();
        vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
        vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());

        VkPipelineInputAssemblyStateCreateInfo inputAssem{};
        inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //set the topology to triangle list (3 vertices per triangle)
        inputAssem.primitiveRestartEnable = VK_FALSE;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swap.extent;

        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &swap.viewport;
        vpState.scissorCount = 1;
        vpState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_TRUE;

        VkPipelineMultisampleStateCreateInfo multiSamp{};
        multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multiSamp.alphaToCoverageEnable = VK_FALSE;
        multiSamp.alphaToOneEnable = VK_FALSE;
        multiSamp.sampleShadingEnable = VK_FALSE;
        multiSamp.minSampleShading = 1.0f;

        VkPipelineDepthStencilStateCreateInfo dStencil{};
        dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dStencil.depthTestEnable = VK_TRUE;
        dStencil.depthWriteEnable = VK_TRUE;
        dStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        dStencil.depthBoundsTestEnable = VK_FALSE;
        dStencil.minDepthBounds = 0.0f;
        dStencil.maxDepthBounds = 1.0f;
        dStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBA{};
        colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBA.blendEnable = VK_FALSE;

        std::array<VkPipelineColorBlendAttachmentState, 4> blendAttachments{};
        blendAttachments.fill(colorBA);

        VkPipelineColorBlendStateCreateInfo colorBS{};
        colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBS.logicOpEnable = VK_FALSE;
        colorBS.logicOp = VK_LOGIC_OP_COPY;
        colorBS.attachmentCount = 4;
        colorBS.pAttachments = blendAttachments.data();

        VkPushConstantRange framePCRange{};
        framePCRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        framePCRange.offset = 0;
        framePCRange.size = sizeof(FramePushConst);

        VkPushConstantRange objectPCRange{};
        objectPCRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        objectPCRange.offset = sizeof(FramePushConst);
        objectPCRange.size = sizeof(ObjectPushConst);

        const std::array<VkDescriptorSetLayout, 2> layouts = { descs.materialTextures.layout.v(), descs.camData.layout.v() };
        const std::array< VkPushConstantRange, 2> ranges = { framePCRange, objectPCRange };

        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.pSetLayouts = layouts.data();
        pipelineLayoutInf.setLayoutCount = static_cast<uint32_t>(layouts.size());
        pipelineLayoutInf.pPushConstantRanges = ranges.data();
        pipelineLayoutInf.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());
        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, deferredPipeline.layout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!!");
        }

        std::array<VkAttachmentDescription, 5> attachments{};
        std::array<VkAttachmentReference, 4> colReferences{};

        for (uint8_t i = 0; i < 4; i++) {
            VkAttachmentDescription& a = attachments[i];
            a.format = deferredData.colorFormats[i]; //format of the color attachment
            a.samples = VK_SAMPLE_COUNT_1_BIT; //number of samples to use for multisampling
            a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //what to do with the data in the attachment before rendering
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE; //what to do with the data in the attachment after rendering
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; //what to do with the stencil data before rendering
            a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; //what to do with the stencil data after rendering
            a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //layout of the image before the render pass starts
            a.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //layout of the image after the render pass ends

            VkAttachmentReference& ref = colReferences[i];
            ref.attachment = i;
            ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        attachments[4].format = depthFormat;
        attachments[4].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[4].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[4].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[4].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[4].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 4;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; //type of pipeline to bind to
        subpass.colorAttachmentCount = 4;
        subpass.pColorAttachments = colReferences.data();
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        //render pass setup: describes the attachments (color & depth) and subpasses used by the pipeline
        VkRenderPassCreateInfo renderPassInf{};
        renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInf.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInf.pAttachments = attachments.data();
        renderPassInf.subpassCount = 1;
        renderPassInf.pSubpasses = &subpass;
        VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, deferredPipeline.renderPass.p());
        if (renderPassResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        //pipeline setup: the data needed to create the pipeline
        VkGraphicsPipelineCreateInfo pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInf.pStages = stages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInf.pVertexInputState = &vertexInputInfo;
        pipelineInf.pInputAssemblyState = &inputAssem;
        pipelineInf.pViewportState = &vpState;
        pipelineInf.pRasterizationState = &rasterizer;
        pipelineInf.pMultisampleState = &multiSamp;
        pipelineInf.pDepthStencilState = &dStencil;
        pipelineInf.pColorBlendState = &colorBS;
        pipelineInf.layout = deferredPipeline.layout.v();
        pipelineInf.renderPass = deferredPipeline.renderPass.v();
        pipelineInf.subpass = 0;
        pipelineInf.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInf.basePipelineIndex = -1;
        VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, deferredPipeline.pipeline.p());
        if (pipelineResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    void createLightingPipeline() {
        VkhShaderModule vertShaderModule = createShaderMod("vertex");
        VkhShaderModule fragShaderModule = createShaderMod("fragment");

        VkPipelineShaderStageCreateInfo vertStage = vkh::createShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
        VkPipelineShaderStageCreateInfo fragStage = vkh::createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertStage, fragStage };

        /*
        // input binding data: tells Vulkan how to read/organize data based on the binding, stride and rate
        VkVertexInputBindingDescription vertBindDesc{};
        vertBindDesc.binding = 0;
        vertBindDesc.stride = sizeof(dvl::Vertex); // number of bytes from one vertex (entry) to the next
        vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // data is per vertex

        // vertex input attributes: defines how the vertex & instance data is structured
        std::vector<VkVertexInputAttributeDescription> attrDesc(2);

        attrDesc[0].binding = 0;
        attrDesc[0].location = 0;
        attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
        attrDesc[0].offset = offsetof(dvl::Vertex, pos);

        // texture coordinates
        attrDesc[1].binding = 0;
        attrDesc[1].location = 1;
        attrDesc[1].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
        attrDesc[1].offset = offsetof(dvl::Vertex, tex);

        // vertex input state: defines the structure of vertex data for the pipeline
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        // specify the vertex binding descriptions
        // tells the gpu how to move through the vertex buffer
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &vertBindDesc;

        // specify the vertex attribute descriptions
        // tells the gpu how to interpret individual attributes for each vertex or instance
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
        vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();
        */

        // no vertex input info
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        // input assembly setup: assembles the vertices into primitives
        // in this case, the primitives represent triangles
        VkPipelineInputAssemblyStateCreateInfo inputAssem{};
        inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //set the topology to triangle list (3 vertices per triangle)
        inputAssem.primitiveRestartEnable = VK_FALSE;

        //scissors setup: defines a region of the framebuffer in which rendering is allowed to happen
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swap.extent;

        // viewport state: defines how the rendered output is mapped to the framebuffer
        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &swap.viewport;
        vpState.scissorCount = 1;
        vpState.pScissors = &scissor;

        //rasterizer setup: transforms 3D primitives into into fragments to display on the screen
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; //if true, fragments that are beyond the near and far planes are clamped
        rasterizer.rasterizerDiscardEnable = VK_FALSE; //if true, geometry never passes through the rasterizer and all primitives would be discarded
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill the area of the poly with fragments
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //polygons with vertices in counter clockwise order, will be considered front facing
        rasterizer.depthBiasEnable = VK_FALSE; //disallows the application of depth bias to fragments

        // multisampling setup: samples multiple points in each pixel and combines them to reduce jagged and blunt edges
        VkPipelineMultisampleStateCreateInfo multiSamp{};
        multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; //number of samples to use per fragment (1 = no mulisampling)
        multiSamp.alphaToCoverageEnable = VK_TRUE; //enables alpha-to-coverage, which uses the alpha component to determine the sample coverage
        multiSamp.alphaToOneEnable = VK_FALSE; //if enabled, forces the alpha component of the color attachment to 1
        multiSamp.sampleShadingEnable = VK_FALSE;// if enabled, would force per sample shading instad of per fragment shading
        multiSamp.minSampleShading = 1.0f; //min fraction for sample shading; closer to one is smoother

        //depth and stencil testing setup: allows for fragments to be discarded based on depth and stencil values
        VkPipelineDepthStencilStateCreateInfo dStencil{};
        dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dStencil.depthTestEnable = VK_TRUE; //enable depth testing
        dStencil.depthWriteEnable = VK_TRUE; //enable writing to the depth buffer
        dStencil.depthCompareOp = VK_COMPARE_OP_LESS; //fragments with smaller depth values pass the depth test
        dStencil.depthBoundsTestEnable = VK_FALSE; //if true, discards fragments whose depth values fall outside the min and max bounds
        dStencil.minDepthBounds = 0.0f; //min depth bound
        dStencil.maxDepthBounds = 1.0f; //max depth bound
        dStencil.stencilTestEnable = VK_FALSE; // disable stencil testing

        //color blend attachment: tells the gpu how the outputted color from the frag shader will be combined with the data in the framebuffer
        VkPipelineColorBlendAttachmentState colorBA{};
        colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; //color channels to apply the blending operation to
        colorBA.blendEnable = VK_FALSE; //disable blending

        /*
        //blending factors for color channels (src = color being output, dst= color already in the framebuffer)
        colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBA.colorBlendOp = VK_BLEND_OP_ADD;

        //blending factors for the alpha channel (src = alpha being output, dst= alpha already in the framebuffer)
        colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBA.alphaBlendOp = VK_BLEND_OP_ADD;
        */

        // color blend state: global pipeline blend settings
        VkPipelineColorBlendStateCreateInfo colorBS{};
        colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBS.logicOpEnable = VK_FALSE; //disable logic operations for color blending
        colorBS.logicOp = VK_LOGIC_OP_COPY;
        colorBS.attachmentCount = 1; //number of color blend attachments
        colorBS.pAttachments = &colorBA;

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(FramePushConst);

        const std::array<VkDescriptorSetLayout, 5> layouts = { descs.deferred.layout.v(), descs.lights.layout.v(), descs.shadowmaps.layout.v(), descs.camData.layout.v(), descs.camDepth.layout.v() };

        // pipeline layout setup: defines the connection between shader stages and resources
        // this data includes: descriptorsets and push constants
        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.pSetLayouts = layouts.data();
        pipelineLayoutInf.setLayoutCount = static_cast<uint32_t>(layouts.size());
        pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
        pipelineLayoutInf.pushConstantRangeCount = 1;

        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, lightingPipeline.layout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!!");
        }

        // color attachment: specifies the properties of the color image used in the render pass
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swap.imageFormat; //format of the color attachment
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; //number of samples to use for multisampling
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //what to do with the data in the attachment before rendering
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; //what to do with the data in the attachment after rendering
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; //what to do with the stencil data before rendering
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; //what to do with the stencil data after rendering
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //layout of the image before the render pass starts
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //layout of the image after the render pass ends

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0; //index of the attachment description in the attachment descriptions array
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //layout to use for the attachment during the subpass

        // subpass: a part of a renderpass that contains a set of attachments and specifies the context in which they are used
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; //type of pipeline to bind to
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = nullptr;

        //render pass setup: describes the attachments (color & depth) and subpasses used by the pipeline
        VkRenderPassCreateInfo renderPassInf{};
        renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInf.attachmentCount = 1;
        renderPassInf.pAttachments = &colorAttachment;
        renderPassInf.subpassCount = 1;
        renderPassInf.pSubpasses = &subpass;
        VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, lightingPipeline.renderPass.p());
        if (renderPassResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        //pipeline setup: the data needed to create the pipeline
        VkGraphicsPipelineCreateInfo pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInf.pStages = stages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInf.pVertexInputState = &vertexInputInfo;
        pipelineInf.pInputAssemblyState = &inputAssem;
        pipelineInf.pViewportState = &vpState;
        pipelineInf.pRasterizationState = &rasterizer;
        pipelineInf.pMultisampleState = &multiSamp;
        pipelineInf.pDepthStencilState = &dStencil;
        pipelineInf.pColorBlendState = &colorBS;
        pipelineInf.layout = lightingPipeline.layout.v();
        pipelineInf.renderPass = lightingPipeline.renderPass.v();
        pipelineInf.subpass = 0;
        pipelineInf.basePipelineHandle = VK_NULL_HANDLE; // no base pipeline for now
        pipelineInf.basePipelineIndex = -1;
        VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, lightingPipeline.pipeline.p());
        if (pipelineResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    void createShadowPipeline() {
        VkhShaderModule vertShaderModule = createShaderMod("shadow_vert");
        VkhShaderModule fragShaderModule = createShaderMod("shadow_frag");

        VkPipelineShaderStageCreateInfo vertStage = vkh::createShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
        VkPipelineShaderStageCreateInfo fragStage = vkh::createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertStage, fragStage };

        VkVertexInputBindingDescription vertBindDesc{};
        vertBindDesc.binding = 0;
        vertBindDesc.stride = sizeof(dvl::Vertex);
        vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputBindingDescription instanceBindDesc{};
        instanceBindDesc.binding = 1;
        instanceBindDesc.stride = sizeof(ModelInstance);
        instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };

        std::array<VkVertexInputAttributeDescription, 5> attrDesc{};
        // vertex position attribute
        attrDesc[0].binding = 0;
        attrDesc[0].location = 0;
        attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
        attrDesc[0].offset = offsetof(dvl::Vertex, pos);

        // pass the model matrix as a per-instance data
        // seperate the matrix into 4 vec4's so it can be quickly passed and processed
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t index = i + 1;
            attrDesc[index].binding = 1;
            attrDesc[index].location = index;
            attrDesc[index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attrDesc[index].offset = offsetof(ModelInstance, model) + sizeof(float) * 4 * i;
        }

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();
        vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
        vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());

        VkPipelineInputAssemblyStateCreateInfo inputAssem{};
        inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssem.primitiveRestartEnable = VK_FALSE;

        VkViewport shadowVP{}; // shadow viewport
        shadowVP.x = 0.0f;
        shadowVP.y = 0.0f;
        shadowVP.width = static_cast<float>(shadowProps.width);
        shadowVP.height = static_cast<float>(shadowProps.height);
        shadowVP.minDepth = 0.0f;
        shadowVP.maxDepth = 1.0f;

        VkRect2D shadowScissor{}; // shadow scissor
        shadowScissor.offset = { 0, 0 };
        shadowScissor.extent.width = shadowProps.width;
        shadowScissor.extent.height = shadowProps.height;

        VkPipelineViewportStateCreateInfo vpState{}; // viewport state
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &shadowVP;
        vpState.scissorCount = 1;
        vpState.pScissors = &shadowScissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_TRUE;
        rasterizer.depthBiasConstantFactor = 1.75f;
        rasterizer.depthBiasSlopeFactor = 1.75f;
        rasterizer.depthBiasClamp = 0.0f;

        VkPipelineMultisampleStateCreateInfo multiSamp{};
        multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSamp.sampleShadingEnable = VK_FALSE;
        multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // creating the depth stencil state
        VkPipelineDepthStencilStateCreateInfo dStencil{};
        dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dStencil.depthTestEnable = VK_TRUE; //enable depth test
        dStencil.depthWriteEnable = VK_TRUE;
        dStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        dStencil.depthBoundsTestEnable = VK_FALSE;
        dStencil.minDepthBounds = 0.0f;
        dStencil.maxDepthBounds = 1.0f;
        dStencil.stencilTestEnable = VK_FALSE;

        // creating the shadow map render pass
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // create the depth attachment reference
        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 0;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // creatr the subpass
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        // create the render pass
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &depthAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, shadowPipeline.renderPass.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow map render pass!");
        }

        VkPipelineColorBlendStateCreateInfo colorBS{};
        colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBS.attachmentCount = 0;

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(ShadowPushConst);

        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.pSetLayouts = descs.lights.layout.p();
        pipelineLayoutInf.setLayoutCount = 1;
        pipelineLayoutInf.pPushConstantRanges = &pcRange;
        pipelineLayoutInf.pushConstantRangeCount = 1;
        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, shadowPipeline.layout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!!");
        }

        // create the pipeline based off this pipeline and some data from the lighting pipeline
        VkGraphicsPipelineCreateInfo pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInf.pStages = stages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInf.pVertexInputState = &vertexInputInfo;
        pipelineInf.pInputAssemblyState = &inputAssem;
        pipelineInf.pViewportState = &vpState;
        pipelineInf.pRasterizationState = &rasterizer;
        pipelineInf.pMultisampleState = &multiSamp;
        pipelineInf.pDepthStencilState = &dStencil;
        pipelineInf.pColorBlendState = &colorBS;
        pipelineInf.layout = shadowPipeline.layout.v();
        pipelineInf.renderPass = shadowPipeline.renderPass.v();
        pipelineInf.subpass = 0;
        pipelineInf.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, shadowPipeline.pipeline.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow map pipeline!");
        }
    }

    void createSkyboxPipeline() {
        VkhShaderModule vertShaderModule = createShaderMod("sky_vert");
        VkhShaderModule fragShaderModule = createShaderMod("sky_frag");

        VkPipelineShaderStageCreateInfo vertStage = vkh::createShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
        VkPipelineShaderStageCreateInfo fragStage = vkh::createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertStage, fragStage };

        VkVertexInputBindingDescription bindDesc{};
        bindDesc.binding = 0;
        bindDesc.stride = sizeof(dml::vec3); // the stride is the size of vec3
        bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrDesc{};
        attrDesc.binding = 0;
        attrDesc.location = 0;
        attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
        attrDesc.offset = 0;

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexAttributeDescriptions = &attrDesc;
        vertexInputInfo.vertexAttributeDescriptionCount = 1;

        VkPipelineInputAssemblyStateCreateInfo inputAssem{};
        inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssem.primitiveRestartEnable = VK_FALSE;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swap.extent;
        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &swap.viewport;
        vpState.scissorCount = 1;
        vpState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multiSamp{};
        multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSamp.sampleShadingEnable = VK_FALSE;
        multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multiSamp.minSampleShading = 1.0f;
        multiSamp.pSampleMask = nullptr;
        multiSamp.alphaToCoverageEnable = VK_FALSE;
        multiSamp.alphaToOneEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo dStencil{};
        dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dStencil.depthTestEnable = VK_TRUE; //enable depth test
        dStencil.depthWriteEnable = VK_FALSE; // dont write to the depth buffer
        dStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        dStencil.depthBoundsTestEnable = VK_FALSE;
        dStencil.minDepthBounds = 0.0f;
        dStencil.maxDepthBounds = 1.0f;
        dStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBA{};
        colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBA.blendEnable = VK_FALSE; // disable blending for the skybox
        VkPipelineColorBlendStateCreateInfo colorBS{};
        colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBS.logicOpEnable = VK_FALSE;
        colorBS.logicOp = VK_LOGIC_OP_COPY;
        colorBS.attachmentCount = 1;
        colorBS.pAttachments = &colorBA;

        VkPushConstantRange framePCRange{};
        framePCRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        framePCRange.offset = 0;
        framePCRange.size = sizeof(FramePushConst);

        const std::array<VkDescriptorSetLayout, 2> layouts = { descs.known.layout.v(), descs.camData.layout.v() };

        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.pSetLayouts = layouts.data();
        pipelineLayoutInf.setLayoutCount = static_cast<uint32_t>(layouts.size());
        pipelineLayoutInf.pPushConstantRanges = &framePCRange;
        pipelineLayoutInf.pushConstantRangeCount = 1;
        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, skybox.pipelineLayout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout for skybox!");
        }

        VkGraphicsPipelineCreateInfo pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInf.pStages = stages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInf.pVertexInputState = &vertexInputInfo;
        pipelineInf.pInputAssemblyState = &inputAssem;
        pipelineInf.pViewportState = &vpState;
        pipelineInf.pRasterizationState = &rasterizer;
        pipelineInf.pMultisampleState = &multiSamp;
        pipelineInf.pDepthStencilState = &dStencil;
        pipelineInf.pColorBlendState = &colorBS;
        pipelineInf.layout = skybox.pipelineLayout.v();
        pipelineInf.renderPass = lightingPipeline.renderPass.v();
        pipelineInf.subpass = 0;
        VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, skybox.pipeline.p());
        if (pipelineResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline for skybox!");
        }
    }

    void createWBOITPipeline() {
        VkhShaderModule vertShaderModule = createShaderMod("wboit_vert");
        VkhShaderModule fragShaderModule = createShaderMod("wboit_frag");

        VkPipelineShaderStageCreateInfo vertStage = vkh::createShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
        VkPipelineShaderStageCreateInfo fragStage = vkh::createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertStage, fragStage };

        VkVertexInputBindingDescription vertBindDesc{};
        vertBindDesc.binding = 0;
        vertBindDesc.stride = sizeof(dvl::Vertex);
        vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputBindingDescription instanceBindDesc{};
        instanceBindDesc.binding = 1;
        instanceBindDesc.stride = sizeof(ModelInstance);
        instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };
        std::array<VkVertexInputAttributeDescription, 9> attrDesc{};

        attrDesc[0].binding = 0;
        attrDesc[0].location = 0;
        attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
        attrDesc[0].offset = offsetof(dvl::Vertex, pos);

        // texture coordinates
        attrDesc[1].binding = 0;
        attrDesc[1].location = 1;
        attrDesc[1].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
        attrDesc[1].offset = offsetof(dvl::Vertex, tex);

        // normal
        attrDesc[2].binding = 0;
        attrDesc[2].location = 2;
        attrDesc[2].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for normal
        attrDesc[2].offset = offsetof(dvl::Vertex, normal);

        // tangents
        attrDesc[3].binding = 0;
        attrDesc[3].location = 3;
        attrDesc[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for tangent
        attrDesc[3].offset = offsetof(dvl::Vertex, tangent);

        // pass the model matrix as a per-instance data
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t index = 4 + i;
            attrDesc[index].binding = 1;
            attrDesc[index].location = index;
            attrDesc[index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attrDesc[index].offset = offsetof(ModelInstance, model) + sizeof(float) * 4 * i;
        }

        attrDesc[8].binding = 1;
        attrDesc[8].location = 8;
        attrDesc[8].format = VK_FORMAT_R32_UINT; // 1 uint32_t
        attrDesc[8].offset = offsetof(ModelInstance, render);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();
        vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
        vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());

        VkPipelineInputAssemblyStateCreateInfo inputAssem{};
        inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssem.primitiveRestartEnable = VK_FALSE;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swap.extent;
        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &swap.viewport;
        vpState.scissorCount = 1;
        vpState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multiSamp{};
        multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSamp.sampleShadingEnable = VK_FALSE;
        multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multiSamp.minSampleShading = 1.0f;
        multiSamp.pSampleMask = nullptr;
        multiSamp.alphaToCoverageEnable = VK_FALSE;
        multiSamp.alphaToOneEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo dStencil{};
        dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dStencil.depthTestEnable = VK_TRUE; //enable depth test
        dStencil.depthWriteEnable = VK_FALSE; //dont write to the depth buffer
        dStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        dStencil.depthBoundsTestEnable = VK_FALSE;
        dStencil.minDepthBounds = 0.0f;
        dStencil.maxDepthBounds = 1.0f;
        dStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBA{};
        colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBA.blendEnable = VK_TRUE;
        colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBA.colorBlendOp = VK_BLEND_OP_ADD;
        colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBA.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBS{};
        colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBS.logicOpEnable = VK_FALSE;
        colorBS.logicOp = VK_LOGIC_OP_COPY;
        colorBS.attachmentCount = 1;
        colorBS.pAttachments = &colorBA;

        // weighted color
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        // define the render pass
        VkRenderPassCreateInfo renderPassInf{};
        renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInf.attachmentCount = 1;
        renderPassInf.pAttachments = &colorAttachment;
        renderPassInf.subpassCount = 1;
        renderPassInf.pSubpasses = &subpass;
        VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, wboitPipeline.renderPass.p());
        if (renderPassResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        VkPushConstantRange framePCRange{};
        framePCRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        framePCRange.offset = 0;
        framePCRange.size = sizeof(FramePushConst);

        VkPushConstantRange objectPCRange{};
        objectPCRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        objectPCRange.offset = sizeof(FramePushConst);
        objectPCRange.size = sizeof(LightPushConst) + sizeof(ObjectPushConst);

        const std::array<VkDescriptorSetLayout, 5> layouts = { descs.materialTextures.layout.v(), descs.lights.layout.v(), descs.shadowmaps.layout.v(), descs.camData.layout.v(), descs.camDepth.layout.v() };
        const std::array< VkPushConstantRange, 2> ranges = { framePCRange, objectPCRange };

        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.pSetLayouts = layouts.data();
        pipelineLayoutInf.setLayoutCount = static_cast<uint32_t>(layouts.size());
        pipelineLayoutInf.pPushConstantRanges = ranges.data();
        pipelineLayoutInf.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());

        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, wboitPipeline.layout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout for WBOIT!!");
        }

        VkGraphicsPipelineCreateInfo pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInf.pStages = stages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInf.pVertexInputState = &vertexInputInfo;
        pipelineInf.pInputAssemblyState = &inputAssem;
        pipelineInf.pViewportState = &vpState;
        pipelineInf.pRasterizationState = &rasterizer;
        pipelineInf.pMultisampleState = &multiSamp;
        pipelineInf.pDepthStencilState = &dStencil;
        pipelineInf.pColorBlendState = &colorBS;
        pipelineInf.layout = wboitPipeline.layout.v();
        pipelineInf.renderPass = wboitPipeline.renderPass.v();
        pipelineInf.subpass = 0;
        VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, wboitPipeline.pipeline.p());
        if (pipelineResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline for WBOIT!");
        }
    }

    void createCompositionPipeline() {
        std::string vert;
        std::string frag;
        if (rtEnabled) {
            vert = "rt_present_vert";
            frag = "rt_present_frag";
        }
        else {
            vert = "composition_vert";
            frag = "composition_frag";
        }

        VkhShaderModule vertShaderModule = createShaderMod(vert);
        VkhShaderModule fragShaderModule = createShaderMod(frag);

        VkPipelineShaderStageCreateInfo vertStage = vkh::createShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
        VkPipelineShaderStageCreateInfo fragStage = vkh::createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertStage, fragStage };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssem{};
        inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssem.primitiveRestartEnable = VK_FALSE;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swap.extent;
        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &swap.viewport;
        vpState.scissorCount = 1;
        vpState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE; // no culling
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multiSamp{};
        multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSamp.rasterizationSamples = compositionSampleCount;
        multiSamp.alphaToCoverageEnable = VK_FALSE;
        multiSamp.alphaToOneEnable = VK_FALSE;
        multiSamp.sampleShadingEnable = VK_TRUE;
        multiSamp.minSampleShading = 0.2f;

        VkPipelineDepthStencilStateCreateInfo dStencil{};
        dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dStencil.depthTestEnable = VK_FALSE; // no depth write or test
        dStencil.depthWriteEnable = VK_FALSE;
        dStencil.depthBoundsTestEnable = VK_FALSE;
        dStencil.minDepthBounds = 0.0f;
        dStencil.maxDepthBounds = 1.0f;
        dStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBA{};
        colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBA.blendEnable = VK_TRUE;
        colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBA.colorBlendOp = VK_BLEND_OP_ADD;
        colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBA.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBS{};
        colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBS.logicOpEnable = VK_FALSE;
        colorBS.logicOp = VK_LOGIC_OP_COPY;
        colorBS.attachmentCount = 1;
        colorBS.pAttachments = &colorBA;

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swap.imageFormat;
        colorAttachment.samples = compositionSampleCount;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorResolve{};
        colorResolve.format = swap.imageFormat;
        colorResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorResolveAttachmentRef{};
        colorResolveAttachmentRef.attachment = 1;
        colorResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pResolveAttachments = &colorResolveAttachmentRef;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, colorResolve };
        VkRenderPassCreateInfo renderPassInf{};
        renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInf.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInf.pAttachments = attachments.data();
        renderPassInf.subpassCount = 1;
        renderPassInf.pSubpasses = &subpass;
        VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, compPipeline.renderPass.p());
        if (renderPassResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.size = sizeof(FramePushConst);
        pcRange.offset = 0;

        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.setLayoutCount = 1;
        pipelineLayoutInf.pSetLayouts = rtEnabled ? descs.rtTex.layout.p() : descs.compTextures.layout.p();
        pipelineLayoutInf.pPushConstantRanges = &pcRange;
        pipelineLayoutInf.pushConstantRangeCount = 1;

        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, compPipeline.layout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout for composition!!");
        }

        VkGraphicsPipelineCreateInfo pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInf.pStages = stages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInf.pVertexInputState = &vertexInputInfo;
        pipelineInf.pInputAssemblyState = &inputAssem;
        pipelineInf.pViewportState = &vpState;
        pipelineInf.pRasterizationState = &rasterizer;
        pipelineInf.pMultisampleState = &multiSamp;
        pipelineInf.pDepthStencilState = &dStencil;
        pipelineInf.pColorBlendState = &colorBS;
        pipelineInf.layout = compPipeline.layout.v();
        pipelineInf.renderPass = compPipeline.renderPass.v();
        pipelineInf.subpass = 0;
        VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, compPipeline.pipeline.p());
        if (pipelineResult != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline for the composition pass!");
        }
    }

    void createRayTracingPipeline() {
        const size_t numShaders = 5;

        std::vector<std::string> shaderNames;
        shaderNames.push_back("gen");
        shaderNames.push_back("miss");
        shaderNames.push_back("shadowmiss");
        shaderNames.push_back("closehit");
        shaderNames.push_back("shadowhit");

        std::vector<VkShaderStageFlagBits> shaderStageFlagBits;;
        shaderStageFlagBits.push_back(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
        shaderStageFlagBits.push_back(VK_SHADER_STAGE_MISS_BIT_KHR);
        shaderStageFlagBits.push_back(VK_SHADER_STAGE_MISS_BIT_KHR);
        shaderStageFlagBits.push_back(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        shaderStageFlagBits.push_back(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

        // populate the shader module and shader stages data
        std::vector<VkhShaderModule> shaderModules;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages{};
        for (uint8_t i = 0; i < numShaders; i++) {
            shaderModules.push_back(createShaderMod(shaderNames[i]));
            shaderStages.push_back(vkh::createShaderStage(shaderStageFlagBits[i], shaderModules[i]));
        }

        std::array<VkRayTracingShaderGroupCreateInfoKHR, numShaders> shaderGroups{};

        // populate the shader group data
        for (uint8_t i = 0; i < numShaders; i++) {
            shaderGroups[i].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            shaderGroups[i].anyHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroups[i].closestHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
        }

        // ray generation group
        shaderGroups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroups[0].generalShader = 0; // ray gen index

        // ray miss group
        shaderGroups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroups[1].generalShader = 1; // ray miss index

        // shadow miss group
        shaderGroups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroups[2].generalShader = 2;

        // ray hit group
        shaderGroups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroups[3].generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroups[3].closestHitShader = 3;

        // shadow hit group
        shaderGroups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroups[4].generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroups[4].closestHitShader = 4;

        VkPushConstantRange genPCRange{};
        genPCRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        genPCRange.offset = 0;
        genPCRange.size = sizeof(FramePushConst);

        VkPushConstantRange chPCRange{};
        chPCRange.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        chPCRange.offset = sizeof(FramePushConst);
        chPCRange.size = sizeof(LightPushConst);

        // create the pipeline layoyut
        const std::array<VkDescriptorSetLayout, 7> layouts = { descs.materialTextures.layout.v(), descs.lights.layout.v(), descs.known.layout.v(), descs.camData.layout.v(), descs.rtTex.layout.v(), descs.tlas.layout.v(), descs.texIndices.layout.v() };
        const std::array< VkPushConstantRange, 2> ranges = { genPCRange, chPCRange };

        VkPipelineLayoutCreateInfo pipelineLayoutInf{};
        pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInf.pSetLayouts = layouts.data();
        pipelineLayoutInf.setLayoutCount = static_cast<uint32_t>(layouts.size());
        pipelineLayoutInf.pPushConstantRanges = ranges.data();
        pipelineLayoutInf.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());
        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, rtPipeline.layout.p());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create raytracing pipeline layout!!");
        }

        // create the pipeline
        VkRayTracingPipelineCreateInfoKHR pipelineInf{};
        pipelineInf.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        pipelineInf.maxPipelineRayRecursionDepth = 4;
        pipelineInf.pStages = shaderStages.data();
        pipelineInf.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInf.pGroups = shaderGroups.data();
        pipelineInf.groupCount = static_cast<uint32_t>(shaderGroups.size());
        pipelineInf.layout = rtPipeline.layout.v();
        if (vkhfp::vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, rtPipeline.pipeline.p()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create ray tracing pipeline!!");
        }
    }

    void setupPipelines(bool shadow) {
        if (rtEnabled) {
            rtPipeline.reset();
            createRayTracingPipeline();
        }
        else {
            deferredPipeline.reset();
            skybox.resetPipeline();
            lightingPipeline.reset();
            wboitPipeline.reset();

            createDeferredPipeline();
            createLightingPipeline();
            createSkyboxPipeline();
            if (shadow) {
                shadowPipeline.reset();
                createShadowPipeline();
            }
            createWBOITPipeline();
        }

        compPipeline.reset();
        createCompositionPipeline();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    static void check_vk_result(VkResult err) { // used to debug imgui errors that have to do with vulkan 
        if (err == 0)
            return;
        std::cerr << "VkResult is " << err << " in " << __FILE__ << " at line " << __LINE__ << "\n";
        assert(err == 0); // if true, continue, if false, throw error
    }

    void guiDSLayout() { //descriptor set layout for imgui
        VkDescriptorSetLayoutBinding imguiBinding{};
        imguiBinding.binding = 0;
        imguiBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        imguiBinding.descriptorCount = 1;
        imguiBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // access through the fragment shader

        VkDescriptorSetLayoutCreateInfo imguiLayoutInfo{};
        imguiLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        imguiLayoutInfo.bindingCount = 1;
        imguiLayoutInfo.pBindings = &imguiBinding;

        if (vkCreateDescriptorSetLayout(device, &imguiLayoutInfo, nullptr, imguiDescriptorSetLayout.p()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create ImGui descriptor set layout!");
        }
    }

    void guiDSPool() { // descriptor pool for imgui
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; // descriptor sets can be freed individually
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, imguiDescriptorPool.p()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Imgui descriptor pool!");
        }
    }

    void imguiSetup() {
        // descriptor set creation for imgui
        guiDSLayout();
        guiDSPool();

        // imgui setup:
        uint32_t graphicsQueueFamily = queueFamilyIndices.graphicsFamily.value();
        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.Instance = instance;
        initInfo.PhysicalDevice = physicalDevice;
        initInfo.Device = device;
        initInfo.QueueFamily = graphicsQueueFamily;
        initInfo.Queue = graphicsQueue;
        initInfo.PipelineCache = VK_NULL_HANDLE; // no pipeline cache for now
        initInfo.DescriptorPool = imguiDescriptorPool.v();
        initInfo.Allocator = VK_NULL_HANDLE;
        initInfo.MinImageCount = swap.imageCount;
        initInfo.ImageCount = swap.imageCount;
        initInfo.CheckVkResultFn = check_vk_result; // function to check vulkan results
        initInfo.MSAASamples = compositionSampleCount;

        ImGui_ImplVulkan_Init(&initInfo, compPipeline.renderPass.v());

        // upload fonts, etc:
        VkhCommandPool guiCommandPool = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        VkhCommandBuffer guiCommandBuffer = vkh::beginSingleTimeCommands(guiCommandPool);
        ImGui_ImplVulkan_CreateFontsTexture(guiCommandBuffer.v());
        vkh::endSingleTimeCommands(guiCommandBuffer, guiCommandPool, graphicsQueue);
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }

    void imguiCleanup() {
        ImGui_ImplVulkan_Shutdown();
        ImGui::DestroyContext();
    }

    void ImguiRenderFrame(VkhCommandBuffer& commandBuffer) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const float p = 10.0f;

        float x = static_cast<float>(swap.extent.width) - p;
        float y = p;
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always, ImVec2(1.0f, 0.0f));

        // window flags
        ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoTitleBar;

        // style settings
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 11.0f);
        ImGui::PushFont(largeFont);

        // text to display
        std::vector<std::string> text;
        text.push_back("FPS: " + std::to_string(fps));
        text.push_back("Objects: " + std::to_string(objects.size()));
        text.push_back("Lights: " + std::to_string(lights.size()));
        text.push_back("Raytracing: " + std::string(rtEnabled ? "ON" : "OFF"));

        // render the frame
        if (ImGui::Begin("Info", nullptr, flags)) {
            for (const auto& t : text) {
                ImGui::TextUnformatted(t.c_str());
            }
        }

        ImGui::End();
        ImGui::PopFont();
        ImGui::PopStyleVar(1);

        // render the imgui frame and draw imgui's commands into the command buffer
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer.v());
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    void createSBT() {
        const uint32_t shaderGroupCount = 5;

        // the size of a single shader group handle (in bytes)
        // shader group handles tell the gpu where to find specific shaders
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;

        // the alignment requirement for the shader group handles in the sbt
        // it ensures each group is properly aligned for maximum efficiency
        uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;

        // the alignment of the shader record data inside the SBT
        // the shader record data holds the shader group handle along with other per-shader data
        uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;

        sbt.entryS = baseAlignment;
        sbt.size = sbt.entryS * shaderGroupCount; // the total size of the sbt

        // create the sbt buffer
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateFlags memAllocF = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        vkh::createBuffer(sbt.buffer, sbt.mem, sbt.size, usage, memFlags, memAllocF);

        // get the shader handles
        std::vector<uint8_t> shaderHandles(handleSize * shaderGroupCount);
        vkhfp::vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline.pipeline.v(), 0, shaderGroupCount, shaderHandles.size(), shaderHandles.data());

        void* data;
        vkMapMemory(device, sbt.mem.v(), 0, sbt.size, 0, &data);
        uint8_t* d = reinterpret_cast<uint8_t*>(data);

        uint32_t dataOffset = 0;
        uint32_t handleOffset = 0;
        // copy the data of each shader group handle into the sbt buffer
        for (uint32_t i = 0; i < shaderGroupCount; i++) {
            std::memcpy(d + dataOffset, shaderHandles.data() + handleOffset, handleSize);
            dataOffset += static_cast<uint32_t>(sbt.entryS);
            handleOffset += handleSize;
        }

        vkUnmapMemory(device, sbt.mem.v());

        VkDeviceAddress sbtAddr = vkh::bufferDeviceAddress(sbt.buffer);

        // ray gen region
        sbt.raygenR.deviceAddress = sbtAddr;
        sbt.raygenR.stride = sbt.entryS;
        sbt.raygenR.size = sbt.entryS;

        // ray miss region
        sbt.missR.deviceAddress = sbtAddr + sbt.entryS;
        sbt.missR.stride = sbt.entryS;
        sbt.missR.size = sbt.entryS * 2;

        // ray hit region
        sbt.hitR.deviceAddress = sbtAddr + (3 * sbt.entryS);
        sbt.hitR.stride = sbt.entryS;
        sbt.hitR.size = sbt.entryS * 2;

        sbt.callR.deviceAddress = 0;
        sbt.callR.stride = 0;
        sbt.callR.size = 0;
    }

    void createBLAS(vkh::BufData bufferData, size_t index) {
        VkhAccelerationStructure blas{};
        uint32_t primitiveCount = bufferData.indexCount / 3;

        // get the device addresses (location of the data on the device) of the vertex and index buffers
        // the stride and offset are used to go to the starting point of the mesh
        // this allows the data within the gpu to be accessed very efficiently
        VkDeviceAddress vertexAddress = vkh::bufferDeviceAddress(vertBuffer) + (bufferData.vertexOffset * sizeof(dvl::Vertex));
        VkDeviceAddress indexAddress = vkh::bufferDeviceAddress(indBuffer) + (bufferData.indexOffset * sizeof(uint32_t));

        // acceleration structure geometry - specifies the device addresses and data inside of the vertex and index buffers
        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = 0; // no geometry flags are set
        geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = vertexAddress;
        geometry.geometry.triangles.vertexStride = sizeof(dvl::Vertex);
        geometry.geometry.triangles.maxVertex = bufferData.vertexCount;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geometry.geometry.triangles.indexData.deviceAddress = indexAddress;

        VkBuildAccelerationStructureFlagsKHR accelerationFlags = 0;
        accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; // allows the blas to be compacted
        accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; // optimizes the blas for faster path tracing

        // BLAS build info - specifies the acceleration structure type, the flags, and the geometry
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.flags = accelerationFlags;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        // size requirements for the BLAS - the total size of the acceleration structure, taking into account the amount of primitives, etc
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkhfp::vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

        // create a buffer for the BLAS - the buffer used in the creation of the blas
        VkhBuffer blasBuffer;
        VkhDeviceMemory blasMem;
        VkBufferUsageFlags blasUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        vkh::createDeviceLocalBuffer(blasBuffer, blasMem, sizeInfo.accelerationStructureSize, blasUsage, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        // create the BLAS
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        createInfo.buffer = blasBuffer.v();
        createInfo.size = sizeInfo.accelerationStructureSize;
        vkhfp::vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, blas.p());

        // scratch buffer - used to create space for intermediate data thats used when building the BLAS
        VkhBuffer blasScratchBuffer;
        VkhDeviceMemory blasScratchMem;
        VkBufferUsageFlags scratchUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        vkh::createDeviceLocalBuffer(blasScratchBuffer, blasScratchMem, sizeInfo.buildScratchSize, scratchUsage, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        // build range info - specifies the primitive count and offsets for the blas
        VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.primitiveCount = primitiveCount;
        buildRangeInfo.primitiveOffset = 0;
        buildRangeInfo.transformOffset = 0;
        buildRangeInfo.firstVertex = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &buildRangeInfo;

        // set the dst of the build info to be the blas and add the scratch buffer address
        buildInfo.dstAccelerationStructure = blas.v();
        buildInfo.scratchData.deviceAddress = vkh::bufferDeviceAddress(blasScratchBuffer);

        // build and populate the BLAS with the geometry data
        VkhCommandBuffer commandBufferB = vkh::beginSingleTimeCommands(commandPool);
        vkhfp::vkCmdBuildAccelerationStructuresKHR(commandBufferB.v(), 1, &buildInfo, &pBuildRangeInfo);
        vkh::endSingleTimeCommands(commandBufferB, commandPool, graphicsQueue);

        // create a query pool used to store the size of the compacted BLAS
        VkhQueryPool queryPool;
        VkQueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        queryPoolInfo.queryCount = 1;
        vkCreateQueryPool(device, &queryPoolInfo, nullptr, queryPool.p());

        // query the size of the BLAS by writing its properties to the query pool
        // the data becomes avaible after submitting the command buffer
        VkhCommandBuffer commandBufferQ = vkh::beginSingleTimeCommands(commandPool);
        vkCmdResetQueryPool(commandBufferQ.v(), queryPool.v(), 0, 1);
        vkhfp::vkCmdWriteAccelerationStructuresPropertiesKHR(commandBufferQ.v(), 1, blas.p(), VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool.v(), 0);
        vkh::endSingleTimeCommands(commandBufferQ, commandPool, graphicsQueue);

        // get the compacted size from the query pool
        VkDeviceSize compactedSize = 0;
        vkGetQueryPoolResults(device, queryPool.v(), 0, 1, sizeof(VkDeviceSize), &compactedSize, sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

        // create a buffer for the compacted BLAS
        VkBufferUsageFlags compUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        vkh::createDeviceLocalBuffer(BLAS[index].compBuffer, BLAS[index].compMem, compactedSize, compUsage, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        // create the compacted BLAS
        VkAccelerationStructureCreateInfoKHR compactedCreateInfo{};
        compactedCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        compactedCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        compactedCreateInfo.buffer = BLAS[index].compBuffer.v();
        compactedCreateInfo.size = compactedSize;
        vkhfp::vkCreateAccelerationStructureKHR(device, &compactedCreateInfo, nullptr, BLAS[index].blas.p());

        // the info for the copying of the original blas to the compacted blas
        VkCopyAccelerationStructureInfoKHR copyInfo{};
        copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        copyInfo.src = blas.v();
        copyInfo.dst = BLAS[index].blas.v();

        // copy the original BLAS to the compacted one
        VkhCommandBuffer commandBufferC = vkh::beginSingleTimeCommands(commandPool);
        vkhfp::vkCmdCopyAccelerationStructureKHR(commandBufferC.v(), &copyInfo);
        vkh::endSingleTimeCommands(commandBufferC, commandPool, graphicsQueue);
    }

    void createTLAS(TlasData& t) {
        if (t.as.valid()) t.as.reset();

        // create a buffer to hold all of the instances
        VkDeviceSize iSize = cfg::MAX_MODELS * sizeof(VkAccelerationStructureInstanceKHR);
        VkBufferUsageFlags iUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateFlags iMemFlags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        vkh::createAndWriteHostBuffer(t.instanceBuffer, t.instanceBufferMem, meshInstances.data(), iSize, iUsage, iMemFlags);

        uint32_t primitiveCount = static_cast<uint32_t>(meshInstances.size());
        uint32_t primitiveCountMax = cfg::MAX_MODELS;

        // acceleration structure geometry
        VkDeviceAddress instanceAddress = vkh::bufferDeviceAddress(t.instanceBuffer);
        t.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        t.geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        t.geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        t.geometry.geometry.instances.data.deviceAddress = instanceAddress;
        t.geometry.geometry.instances.arrayOfPointers = VK_FALSE;

        VkBuildAccelerationStructureFlagsKHR accelerationFlags = 0;
        accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; // optimizes the tlas for faster path tracing
        accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR; // allows the tlas to be updated, without having to fully recreate it

        // TLAS build info - specifies the acceleration structure type, the flags, and the geometry
        t.buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        t.buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        t.buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        t.buildInfo.flags = accelerationFlags;
        t.buildInfo.geometryCount = 1;
        t.buildInfo.pGeometries = &t.geometry;

        // size requirements for the TLAS - the total size of the acceleration structure, taking into account the amount of primitives, etc
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkhfp::vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &t.buildInfo, &primitiveCountMax, &sizeInfo);

        VkDeviceSize asSize = sizeInfo.accelerationStructureSize;

        // create a buffer for the TLAS - the buffer used in the creation of the tlas
        VkBufferUsageFlags tlasUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        vkh::createDeviceLocalBuffer(t.buffer, t.mem, asSize, tlasUsage, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        // create the TLAS
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        createInfo.buffer = t.buffer.v();
        createInfo.size = asSize;
        vkhfp::vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, t.as.p());

        // scratch buffer - used to create space for intermediate data thats used when building the TLAS
        VkBufferUsageFlags scratchUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        vkh::createDeviceLocalBuffer(t.scratchBuffer, t.scratchMem, sizeInfo.buildScratchSize, scratchUsage, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        // build range info - specifies the primitive count and offsets for the tlas
        VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.primitiveCount = primitiveCount;
        buildRangeInfo.primitiveOffset = 0;
        buildRangeInfo.transformOffset = 0;
        buildRangeInfo.firstVertex = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &buildRangeInfo;

        // set the dst of the build info to be the tlas and add the scratch buffer address
        t.buildInfo.dstAccelerationStructure = t.as.v();
        t.buildInfo.scratchData.deviceAddress = vkh::bufferDeviceAddress(t.scratchBuffer);

        // build and populate the TLAS
        VkhCommandBuffer commandBufferB = vkh::beginSingleTimeCommands(commandPool);
        vkhfp::vkCmdBuildAccelerationStructuresKHR(commandBufferB.v(), 1, &t.buildInfo, &pBuildRangeInfo);
        vkh::endSingleTimeCommands(commandBufferB, commandPool, graphicsQueue);
    }

    VkTransformMatrixKHR mat4ToVk(const dml::mat4& m) {
        dml::mat4 t = m.transpose();

        VkTransformMatrixKHR result{};
        std::memcpy(&result.matrix, &t.flat, 12 * sizeof(float));
        return result;
    }


    void createMeshInstace(size_t index) {
        VkAccelerationStructureInstanceKHR meshInstance{};
        size_t bufferInd = modelHashToBufferIndex[objects[index]->meshHash];

        // copy the models model matrix into the instance data
        meshInstance.transform = mat4ToVk(objInstanceData.object[index].model);

        VkDeviceAddress blasAddress = vkh::asDeviceAddress(BLAS[bufferInd].blas);

        // populate the instance data
        meshInstance.accelerationStructureReference = blasAddress;
        meshInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        size_t meshInd = uniqueModelIndex[objects[index]->meshHash];
        meshInstance.instanceCustomIndex = static_cast<uint32_t>(meshInd);

        meshInstance.instanceShaderBindingTableRecordOffset = 0;
        meshInstance.mask = 0xFF;
        meshInstances.push_back(meshInstance);
    }

    void updateTLAS(size_t index, bool rebuild) {
        // copy the new data into the instance buffer
        TlasData& t = tlas[index];
        vkh::writeBuffer(t.instanceBufferMem, meshInstances.data(), meshInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));

        // update the instance buffer device address
        t.geometry.geometry.instances.data.deviceAddress = vkh::bufferDeviceAddress(t.instanceBuffer);

        // update the build info
        t.buildInfo.pGeometries = &t.geometry;
        t.buildInfo.mode = rebuild ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        t.buildInfo.srcAccelerationStructure = rebuild ? VK_NULL_HANDLE : t.as.v();
        t.buildInfo.dstAccelerationStructure = t.as.v();

        // update the buildRangeInfo
        VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.primitiveCount = static_cast<uint32_t>(meshInstances.size());
        buildRangeInfo.primitiveOffset = 0;
        buildRangeInfo.transformOffset = 0;
        buildRangeInfo.firstVertex = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &buildRangeInfo;

        // rebuild and populate the TLAS
        VkhCommandBuffer commandBufferB = vkh::beginSingleTimeCommands(commandPool);
        vkhfp::vkCmdBuildAccelerationStructuresKHR(commandBufferB.v(), 1, &t.buildInfo, &pBuildRangeInfo);
        vkh::endSingleTimeCommands(commandBufferB, commandPool, graphicsQueue);
    }

    void updateTLAS(bool changed = false) {
        if (changed) meshInstances.clear();
        for (size_t i = 0; i < objects.size(); i++) {
            if (changed) {
                createMeshInstace(i);
            }
            else {
                VkAccelerationStructureInstanceKHR& meshInstance = meshInstances[i];
                meshInstance.transform = mat4ToVk(objInstanceData.object[i].model);
            }
        }

        if (changed) {
            for (size_t i = 0; i < maxFrames; i++) {
                updateTLAS(i, true);
            }
        }
        else {
            updateTLAS(currentFrame, false);
        }
    }

    void setupAccelerationStructures() {
        BLAS.resize(getUniqueModels());

        for (size_t i = 0; i < objects.size(); i++) {
            size_t modelInd = uniqueModelIndex[objects[i]->meshHash];
            if (modelInd != i) continue; // skip if not the first instance of the model
            size_t bufferInd = modelHashToBufferIndex[objects[i]->meshHash];

            createBLAS(bufferData[bufferInd], bufferInd);
        }

        for (size_t i = 0; i < objects.size(); i++) {
            createMeshInstace(i);
        }

        tlas.resize(maxFrames);
        for (size_t i = 0; i < maxFrames; i++) {
            createTLAS(tlas[i]);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    uint32_t getModelNumHash(size_t hash) { // get the number of models that have the same hash
        uint32_t count = 0;
        for (auto& m : objects) {
            if (m->meshHash == hash) {
                count++;
            }
        }
        return count;
    }

    size_t getUniqueModels() { // get the number of unique models
        std::unordered_set<size_t> uniqueModels;
        for (auto& m : objects) {
            uniqueModels.insert(m->meshHash);
        }
        return uniqueModels.size();
    }

    size_t originalObjectsNameIndex(const std::string& name) {
        for (size_t i = 0; i < originalObjects.size(); i++) {
            if (originalObjects[i]->name == name) {
                return i;
            }
        }

        LOG_WARNING("mesh: " + name + " wasnt found in originalObjects!");
        return 0;
    }

    void getModelIndices(bool getSize) {
        std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a->meshHash < b->meshHash; });

        uniqueModelIndex.clear();
        modelHashToBufferIndex.clear();

        uint32_t index = 0;
        for (size_t i = 0; i < objects.size(); i++) {
            auto& obj = objects[i];
            if (uniqueModelIndex.find(obj->meshHash) == uniqueModelIndex.end()) {
                if (getSize) {
                    vertBufferSize += sizeof(dvl::Vertex) * obj->vertices.size();
                    indBufferSize += sizeof(uint32_t) * obj->indices.size();
                }

                uniqueModelIndex[obj->meshHash] = i;
                modelHashToBufferIndex[obj->meshHash] = index++;
            }
        }
    }

    void createModelBuffers(bool recreate) {
        if (!recreate) bufferData.resize(getUniqueModels());
        getModelIndices(true);

        VkhBuffer stagingVertBuffer;
        VkhDeviceMemory stagingVertBufferMem;

        VkhBuffer stagingIndexBuffer;
        VkhDeviceMemory stagingIndexBufferMem;

        const VkMemoryPropertyFlags stagingMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        // create and map the vertex buffer
        vkh::createBuffer(stagingVertBuffer, stagingVertBufferMem, vertBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingMemFlags, 0);
        char* vertexData;
        vkMapMemory(device, stagingVertBufferMem.v(), 0, vertBufferSize, 0, reinterpret_cast<void**>(&vertexData));
        VkDeviceSize currentVertexOffset = 0;

        // create and map the index buffer
        vkh::createBuffer(stagingIndexBuffer, stagingIndexBufferMem, indBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingMemFlags, 0);
        char* indexData;
        vkMapMemory(device, stagingIndexBufferMem.v(), 0, indBufferSize, 0, reinterpret_cast<void**>(&indexData));
        VkDeviceSize currentIndexOffset = 0;

        for (size_t i = 0; i < objects.size(); i++) {
            size_t modelInd = uniqueModelIndex[objects[i]->meshHash];
            if (modelInd != i) continue; // skip if not the first instance of the model
            size_t bufferInd = modelHashToBufferIndex[objects[i]->meshHash];

            // vertex data
            bufferData[bufferInd].vertexOffset = static_cast<uint32_t>(currentVertexOffset);
            bufferData[bufferInd].vertexCount = static_cast<uint32_t>(objects[modelInd]->vertices.size());
            std::memcpy(vertexData, objects[modelInd]->vertices.data(), bufferData[bufferInd].vertexCount * sizeof(dvl::Vertex));
            vertexData += bufferData[bufferInd].vertexCount * sizeof(dvl::Vertex);
            currentVertexOffset += bufferData[bufferInd].vertexCount;

            // index data
            bufferData[bufferInd].indexOffset = static_cast<uint32_t>(currentIndexOffset);
            bufferData[bufferInd].indexCount = static_cast<uint32_t>(objects[modelInd]->indices.size());
            std::memcpy(indexData, objects[modelInd]->indices.data(), bufferData[bufferInd].indexCount * sizeof(uint32_t));
            indexData += bufferData[bufferInd].indexCount * sizeof(uint32_t);
            currentIndexOffset += bufferData[bufferInd].indexCount;
        }
        vkUnmapMemory(device, stagingVertBufferMem.v());
        vkUnmapMemory(device, stagingIndexBufferMem.v());

        VkBufferUsageFlags vertU = (rtEnabled) ? (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) : (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkBufferUsageFlags indexU = (rtEnabled) ? (VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) : (VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        VkMemoryAllocateFlags vertM = (rtEnabled) ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
        VkMemoryAllocateFlags indexM = (rtEnabled) ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

        vkh::createBuffer(vertBuffer, vertBufferMem, vertBufferSize, vertU, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertM);
        vkh::createBuffer(indBuffer, indBufferMem, indBufferSize, indexU, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexM);

        // copy the vert staging buffer into the dst vert buffer
        vkh::copyBuffer(stagingVertBuffer, vertBuffer, commandPool, graphicsQueue, vertBufferSize);

        // copy the index staging buffer into the dst index buffer
        vkh::copyBuffer(stagingIndexBuffer, indBuffer, commandPool, graphicsQueue, indBufferSize);
    }

    void cloneObject(dml::vec3 pos, const std::string& name, dml::vec3 scale, dml::vec4 rotation) {
        size_t index = originalObjectsNameIndex(name);
        const std::unique_ptr<dvl::Mesh>& other = originalObjects[index];

        dvl::Mesh m;
        m.scale = scale;
        m.position = pos;
        m.rotation = rotation;
        m.meshHash = other->meshHash;
        m.material = other->material;

        dml::mat4 newModel = dml::translate(pos) * dml::rotateQuat(rotation) * dml::scale(scale);
        m.modelMatrix = newModel * other->modelMatrix;
        objects.push_back(std::make_unique<dvl::Mesh>(std::move(m)));
    }

    void summonModel() {
        if (objects.size() + 2 >= cfg::MAX_MODELS) return;

        dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);
        cloneObject(pos, "Soi_SimpleArmor.001_Armour_Cloth_Worn_0", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });
        cloneObject(pos, "Soi_SimpleArmor.001_Armour_Metal_Worn_0", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });

        getModelIndices(false);

        if (rtEnabled) {
            updateTLAS(true);
            getTexIndices();
        }
    }

    void summonLight() {
        if (lights.size() + 1 > cfg::MAX_LIGHTS) return;
        vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);

        dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);
        dml::vec3 target = pos + dml::quatToDir(cam.quat);
        Light l = createLight(pos, target);

        if (!rtEnabled) {
            size_t index = lights.size();
            for (size_t i = 0; i < maxFrames; i++) {
                dvl::Texture s{};
                VkhFramebuffer f{};
                vkh::createImage(s.image, s.memory, shadowProps.width, shadowProps.height, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, s.sampleCount);
                vkh::createImageView(s, vkh::DEPTH);
                vkh::createSampler(s.sampler, s.mipLevels, vkh::DEPTH);
                vkh::createFB(shadowPipeline.renderPass, f, s.imageView.p(), 1, shadowProps.width, shadowProps.height);

                l.shadowMapData.push_back(s);
                l.frameBuffer.push_back(f);

                VkhCommandPool p1 = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
                VkhCommandBuffer c1 = vkh::allocateCommandBuffers(p1);
                shadowMapCommandBuffers.primary.pools.push_back(p1);
                shadowMapCommandBuffers.primary.buffers.push_back(c1);

                VkhCommandPool p2 = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
                VkhCommandBuffer c2 = vkh::allocateCommandBuffers(p2, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
                shadowMapCommandBuffers.secondary.pools.push_back(p2);
                shadowMapCommandBuffers.secondary.buffers.push_back(c2);

                shadowInfos.push_back(vkh::createDSImageInfo(s.imageView, s.sampler));

                index += lights.size();
            }

            lights.push_back(std::make_unique<Light>(l));
            updateLightDS();
        }
        else {
            lights.push_back(std::make_unique<Light>(l));
        }
    }

    void resetScene() {
        vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);

        // remove all non player following lights
        lights.erase(std::remove_if(lights.begin(), lights.end(), [](const std::unique_ptr<Light>& l) { return l && !l->followPlayer; }), lights.end());

        if (!rtEnabled) {
            allocateCommandBuffers(shadowMapCommandBuffers, lights.size() * maxFrames, lights.size() * maxFrames);

            shadowInfos.clear();
            for (size_t i = 0; i < maxFrames; i++) {
                VkDescriptorImageInfo shadowInfo{};
                shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                shadowInfo.imageView = lights.back()->shadowMapData[i].imageView.v();
                shadowInfo.sampler = lights.back()->shadowMapData[i].sampler.v();
                shadowInfos.push_back(shadowInfo);
            }

            updateLightDS();
        }

        objects.clear();
        objects.reserve(originalObjects.size());
        for (const std::unique_ptr<dvl::Mesh>& m : originalObjects) {
            objects.push_back(std::make_unique<dvl::Mesh>(*m));
        }

        getModelIndices(false);

        if (rtEnabled) {
            updateTLAS(true);
            getTexIndices();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void allocateCommandBuffers(CommandBufferSet& cmdBuffers, size_t primaryCount, size_t secondaryCount = 0) {
        cmdBuffers.primary.reserveClear(primaryCount);

        for (size_t i = 0; i < primaryCount; i++) {
            cmdBuffers.primary.pools.push_back(vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));
            cmdBuffers.primary.buffers.push_back(vkh::allocateCommandBuffers(cmdBuffers.primary.pools[i]));
        }

        if (secondaryCount) {
            cmdBuffers.secondary.reserveClear(secondaryCount);

            for (size_t i = 0; i < secondaryCount; i++) {
                cmdBuffers.secondary.pools.push_back(vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));
                cmdBuffers.secondary.buffers.push_back(vkh::allocateCommandBuffers(cmdBuffers.secondary.pools[i], VK_COMMAND_BUFFER_LEVEL_SECONDARY));
            }
        }
    }

    void createCommandBuffers() {
        if (!rtEnabled) {
            allocateCommandBuffers(deferredCommandBuffers, maxFrames, 1);
            allocateCommandBuffers(shadowMapCommandBuffers, lights.size() * maxFrames, lights.size() * maxFrames);
            allocateCommandBuffers(lightingPassCommandBuffers, maxFrames, 0);
            allocateCommandBuffers(wboitCommandBuffers, maxFrames, 0);
        }
        else {
            allocateCommandBuffers(rtCommandBuffers, swap.imageCount, 0);
        }

        allocateCommandBuffers(compCommandBuffers, swap.imageCount, 0);
    }

    void createFrameBuffers(bool shadow) {
        if (!rtEnabled) {
            deferredData.frameBuffer.resize(maxFrames);
            lightingData.frameBuffer.resize(maxFrames);
            wboit.frameBuffer.resize(maxFrames);

            for (size_t i = 0; i < maxFrames; i++) {

                // deferred pass framebuffers
                std::array<VkImageView, 5> attachments{};
                for (size_t j = 0; j < 4; j++) {
                    size_t k = (i * 4) + j;
                    attachments[j] = deferredData.textures[k].imageView.v();
                }

                attachments[4] = deferredData.depth[i].imageView.v();
                vkh::createFB(deferredPipeline.renderPass, deferredData.frameBuffer[i], attachments.data(), 5, swap.extent.width, swap.extent.height);

                // shadowmap framebuffers
                if (shadow) {
                    for (size_t j = 0; j < lights.size(); j++) {
                        VkhFramebuffer fb{};
                        vkh::createFB(shadowPipeline.renderPass, fb, lights[j]->shadowMapData[i].imageView.p(), 1, shadowProps.width, shadowProps.height);
                        lights[j]->frameBuffer.push_back(fb);
                    }
                }

                // lighting pass framebuffer
                vkh::createFB(lightingPipeline.renderPass, lightingData.frameBuffer[i], lightingData.color[i].imageView.p(), 1, swap.extent.width, swap.extent.height);

                // wboit framebuffer
                vkh::createFB(wboitPipeline.renderPass, wboit.frameBuffer[i], wboit.weightedColor[i].imageView.p(), 1, swap.extent.width, swap.extent.height);
            }
        }

        // composition framebuffers
        swap.framebuffers.resize(swap.imageCount);
        for (size_t i = 0; i < swap.imageCount; i++) {
            std::array<VkImageView, 2> attachments = { compTextures[i].imageView.v(), swap.imageViews[i].v() };
            vkh::createFB(compPipeline.renderPass, swap.framebuffers[i], attachments.data(), 2, swap.extent.width, swap.extent.height);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void recordObjectCommandBuffers(VkhCommandBuffer& secondary, const PipelineData& pipe, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, size_t descriptorCount, uint32_t pushConstOffset) {
        const std::array<VkBuffer, 2> vertexBuffersArray = { vertBuffer.v(), objInstanceBuffers[currentFrame].v() };
        const std::array<VkDeviceSize, 2> offsets = { 0, 0 };

        vkCmdBindPipeline(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipeline.v());
        vkCmdBindDescriptorSets(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout.v(), 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

        // bind the vertex and instance buffers
        vkCmdBindVertexBuffers(secondary.v(), 0, 2, vertexBuffersArray.data(), offsets.data());
        vkCmdBindIndexBuffer(secondary.v(), indBuffer.v(), 0, VK_INDEX_TYPE_UINT32);

        uint32_t p = 0;
        for (size_t j = 0; j < objects.size(); j++) {
            size_t uniqueModelInd = uniqueModelIndex[objects[j]->meshHash];
            if (uniqueModelInd == j) { // only process unique models
                // bitfield for which textures exist
                int textureExistence = 0;
                textureExistence |= (objects[j]->material.baseColor.found ? 1 : 0);
                textureExistence |= (objects[j]->material.metallicRoughness.found ? 1 : 0) << 1;
                textureExistence |= (objects[j]->material.normalMap.found ? 1 : 0) << 2;
                textureExistence |= (objects[j]->material.emissiveMap.found ? 1 : 0) << 3;
                textureExistence |= (objects[j]->material.occlusionMap.found ? 1 : 0) << 4;

                ObjectPushConst pushConst{};
                pushConst.textureExist = textureExistence;
                pushConst.texIndex = meshTexStartInd[p];

                vkCmdPushConstants(secondary.v(), pipe.layout.v(), VK_SHADER_STAGE_FRAGMENT_BIT, pushConstOffset, sizeof(ObjectPushConst), &pushConst);

                size_t bufferInd = modelHashToBufferIndex[objects[j]->meshHash];
                uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->meshHash);
                vkCmdDrawIndexed(secondary.v(), bufferData[bufferInd].indexCount, instanceCount,
                    bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, static_cast<uint32_t>(uniqueModelInd));
                p++;
            }
        }
    }

    void recordDeferredCommandBuffers() {
        const std::array<VkDescriptorSet, 2> deferredDS = { descs.materialTextures.set.v(), descs.camData.set.v() };

        std::array<VkClearValue, 5> clearValues{};
        clearValues.fill(VkClearValue{ 0.0f, 0.0f, 0.0f, 1.0f });
        clearValues[4] = VkClearValue{ 1.0f, 0.0f };

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        VkhCommandBuffer& deferredCommandBuffer = deferredCommandBuffers.primary[currentFrame];
        if (vkBeginCommandBuffer(deferredCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = deferredPipeline.renderPass.v();
        renderPassInfo.framebuffer = deferredData.frameBuffer[currentFrame].v();
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swap.extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(deferredCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(deferredCommandBuffer.v(), deferredPipeline.layout.v(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(FramePushConst), &framePushConst);
        recordObjectCommandBuffers(deferredCommandBuffer, deferredPipeline, beginInfo, deferredDS.data(), deferredDS.size(), sizeof(FramePushConst));

        vkCmdEndRenderPass(deferredCommandBuffer.v());
        if (vkEndCommandBuffer(deferredCommandBuffer.v()) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordShadowSecondaryCommandBuffers() {
        VkCommandBufferInheritanceInfo shadowInheritInfo{};
        shadowInheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        shadowInheritInfo.renderPass = shadowPipeline.renderPass.v();
        shadowInheritInfo.framebuffer = VK_NULL_HANDLE;
        shadowInheritInfo.subpass = 0;

        VkCommandBufferBeginInfo shadowBeginInfo{};
        shadowBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        shadowBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        shadowBeginInfo.pInheritanceInfo = &shadowInheritInfo;

        const std::array<VkBuffer, 2> vertexBuffersArray = { vertBuffer.v(), objInstanceBuffers[currentFrame].v() };
        const std::array<VkDeviceSize, 2> offsets = { 0, 0 };

        for (size_t i = 0; i < lights.size(); i++) {
            size_t index = (i * maxFrames) + currentFrame;
            VkCommandBuffer& secondary = shadowMapCommandBuffers.secondary.buffers[index].v();
            if (vkBeginCommandBuffer(secondary, &shadowBeginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording shadow secondary command buffer!");
            }

            vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline.pipeline.v());
            vkCmdBindDescriptorSets(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline.layout.v(), 0, 1, descs.lights.set.p(), 0, nullptr);

            ShadowPushConst shadowPushConst{};
            shadowPushConst.frame = static_cast<int>(currentFrame);
            shadowPushConst.lightIndex = static_cast<int>(i);

            vkCmdPushConstants(secondary, shadowPipeline.layout.v(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConst), &shadowPushConst);

            vkCmdBindVertexBuffers(secondary, 0, 2, vertexBuffersArray.data(), offsets.data());
            vkCmdBindIndexBuffer(secondary, indBuffer.v(), 0, VK_INDEX_TYPE_UINT32);
        }

        // iterate through all objects that cast shadows
        uint32_t p = 0;
        for (size_t j = 0; j < objects.size(); j++) {
            size_t uniqueModelInd = uniqueModelIndex[objects[j]->meshHash];
            if (uniqueModelInd == j) {
                size_t bufferInd = modelHashToBufferIndex[objects[j]->meshHash];
                uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->meshHash);
                for (size_t i = 0; i < lights.size(); i++) {
                    size_t index = (i * maxFrames) + currentFrame;

                    VkCommandBuffer& secondary = shadowMapCommandBuffers.secondary.buffers[index].v();
                    vkCmdDrawIndexed(secondary, bufferData[bufferInd].indexCount, instanceCount, bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, static_cast<uint32_t>(uniqueModelInd));
                }
                p++;
            }
        }

        for (size_t i = 0; i < lights.size(); i++) {
            size_t index = (i * maxFrames) + currentFrame;

            if (vkEndCommandBuffer(shadowMapCommandBuffers.secondary.buffers[index].v()) != VK_SUCCESS) {
                throw std::runtime_error("failed to record shadow secondary command buffer!");
            }
        }
    }

    void recordShadowCommandBuffers() {
        recordShadowSecondaryCommandBuffers();

        shadowCmdTasks.clear();
        shadowCmdTasks.reserve(lights.size());

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        const VkClearValue clearValue = VkClearValue{ 1.0f, 0 };

        for (size_t i = 0; i < lights.size(); i++) {
            shadowCmdTasks.emplace_back(std::async(std::launch::async, [&, i, beginInfo]() {
                size_t index = (i * maxFrames) + currentFrame;

                VkhCommandBuffer& shadowCommandBuffer = shadowMapCommandBuffers.primary.buffers[index];
                VkhCommandBuffer& secondary = shadowMapCommandBuffers.secondary.buffers[index];

                if (vkBeginCommandBuffer(shadowCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
                    throw std::runtime_error("failed to begin recording command buffer!");
                }

                // render pass
                VkRenderPassBeginInfo renderPassInfo{};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = shadowPipeline.renderPass.v();
                renderPassInfo.framebuffer = lights[i]->frameBuffer[currentFrame].v();
                renderPassInfo.renderArea.offset = { 0, 0 };
                renderPassInfo.renderArea.extent = { shadowProps.width, shadowProps.height };
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = &clearValue;
                vkCmdBeginRenderPass(shadowCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

                vkCmdExecuteCommands(shadowCommandBuffer.v(), 1, secondary.p());

                // end the render pass and command buffer
                vkCmdEndRenderPass(shadowCommandBuffer.v());
                if (vkEndCommandBuffer(shadowCommandBuffer.v()) != VK_SUCCESS) {
                    throw std::runtime_error("failed to record command buffer!");
                }
                }));
        }

        for (size_t i = 0; i < lights.size(); i++) {
            shadowCmdTasks[i].wait();
        }
    }

    void recordLightingCommandBuffers() {
        const std::array<VkDescriptorSet, 5> lightingDS = { descs.deferred.set.v(), descs.lights.set.v(), descs.shadowmaps.set.v(), descs.camData.set.v(), descs.camDepth.set.v() };
        const std::array<VkDescriptorSet, 2> skyboxDS = { descs.known.set.v(), descs.camData.set.v() };

        const VkClearValue clearValue = VkClearValue{ 0.18f, 0.3f, 0.30f, 1.0f };
        const VkDeviceSize skyboxOffset = 0;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        VkhCommandBuffer& lightingCommandBuffer = lightingPassCommandBuffers.primary[currentFrame];
        if (vkBeginCommandBuffer(lightingCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = lightingPipeline.renderPass.v();
        renderPassInfo.framebuffer = lightingData.frameBuffer[currentFrame].v();
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swap.extent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValue;

        vkh::transitionImageLayout(lightingCommandBuffer, deferredData.depth[currentFrame].image, depthFormat, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);

        vkCmdBeginRenderPass(lightingCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // skybox
        vkCmdBindPipeline(lightingCommandBuffer.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipeline.v());
        vkCmdBindDescriptorSets(lightingCommandBuffer.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipelineLayout.v(), 0, static_cast<uint32_t>(skyboxDS.size()), skyboxDS.data(), 0, nullptr);
        vkCmdBindVertexBuffers(lightingCommandBuffer.v(), 0, 1, skybox.vertBuffer.p(), &skyboxOffset);
        vkCmdBindIndexBuffer(lightingCommandBuffer.v(), skybox.indBuffer.v(), 0, VK_INDEX_TYPE_UINT32);

        vkCmdPushConstants(lightingCommandBuffer.v(), skybox.pipelineLayout.v(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(FramePushConst), &framePushConst);
        vkCmdDrawIndexed(lightingCommandBuffer.v(), skybox.bufferData.indexCount, 1, skybox.bufferData.indexOffset, skybox.bufferData.vertexOffset, 0);

        // lighting
        vkCmdBindPipeline(lightingCommandBuffer.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipeline.pipeline.v());
        vkCmdBindDescriptorSets(lightingCommandBuffer.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipeline.layout.v(), 0, static_cast<uint32_t>(lightingDS.size()), lightingDS.data(), 0, nullptr);

        vkCmdPushConstants(lightingCommandBuffer.v(), lightingPipeline.layout.v(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FramePushConst), &framePushConst);

        vkCmdDraw(lightingCommandBuffer.v(), 6, 1, 0, 0);
        vkCmdEndRenderPass(lightingCommandBuffer.v());

        if (vkEndCommandBuffer(lightingCommandBuffer.v()) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordWBOITCommandBuffers() {
        const std::array<VkDescriptorSet, 5> wboitDS = { descs.materialTextures.set.v(),  descs.lights.set.v(), descs.shadowmaps.set.v(), descs.camData.set.v(), descs.camDepth.set.v() };
        const std::array<VkClearValue, 3> clearValues = { VkClearValue{0.0f, 0.0f, 0.0f, 1.0f}, VkClearValue{1.0f}, VkClearValue{1.0f, 0} };

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        VkhCommandBuffer& wboitCommandBuffer = wboitCommandBuffers.primary[currentFrame];
        if (vkBeginCommandBuffer(wboitCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = wboitPipeline.renderPass.v();
        renderPassInfo.framebuffer = wboit.frameBuffer[currentFrame].v();
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swap.extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(wboitCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(wboitCommandBuffer.v(), wboitPipeline.layout.v(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(FramePushConst), &framePushConst);

        LightPushConst lightPushConst{};
        lightPushConst.lightCount = static_cast<int>(lights.size());
        lightPushConst.frameCount = static_cast<int>(maxFrames);

        vkCmdPushConstants(wboitCommandBuffer.v(), wboitPipeline.layout.v(), VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(FramePushConst), sizeof(LightPushConst), &lightPushConst);
        recordObjectCommandBuffers(wboitCommandBuffer, wboitPipeline, beginInfo, wboitDS.data(), wboitDS.size(), sizeof(FramePushConst) + sizeof(LightPushConst));

        vkCmdEndRenderPass(wboitCommandBuffer.v());

        if (vkEndCommandBuffer(wboitCommandBuffer.v()) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordCompCommandBuffers() {
        const std::array<VkClearValue, 2> clearValues = { VkClearValue{0.18f, 0.3f, 0.30f, 1.0f}, VkClearValue{1.0f, 0} };

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        VkDescriptorSet* set = (rtEnabled) ? descs.rtTex.set.p() : descs.compTextures.set.p();

        VkhCommandBuffer& compCommandBuffer = compCommandBuffers.primary.buffers[currentFrame];
        if (vkBeginCommandBuffer(compCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = compPipeline.renderPass.v();
        renderPassInfo.framebuffer = swap.framebuffers[swap.imageIndex].v();
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swap.extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(compCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(compCommandBuffer.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, compPipeline.pipeline.v());
        vkCmdBindDescriptorSets(compCommandBuffer.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, compPipeline.layout.v(), 0, 1, set, 0, nullptr);

        vkCmdPushConstants(compCommandBuffer.v(), compPipeline.layout.v(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FramePushConst), &framePushConst);

        vkCmdDraw(compCommandBuffer.v(), 6, 1, 0, 0);

        ImguiRenderFrame(compCommandBuffer);

        vkCmdEndRenderPass(compCommandBuffer.v());
        if (vkEndCommandBuffer(compCommandBuffer.v()) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordRTCommandBuffers() {
        const std::array<VkDescriptorSet, 7> ds = { descs.materialTextures.set.v(), descs.lights.set.v(), descs.known.set.v(), descs.camData.set.v(), descs.rtTex.set.v(), descs.tlas.set.v(), descs.texIndices.set.v() };

        VkCommandBufferInheritanceInfo inheritInfo{};
        inheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritInfo.renderPass = VK_NULL_HANDLE;
        inheritInfo.framebuffer = VK_NULL_HANDLE;
        inheritInfo.subpass = 0;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = &inheritInfo;

        VkCommandBuffer& commandBuffer = rtCommandBuffers.primary[currentFrame].v();
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording rt command buffer!");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.pipeline.v());
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline.layout.v(), 0, static_cast<uint32_t>(ds.size()), ds.data(), 0, nullptr);

        vkCmdPushConstants(commandBuffer, rtPipeline.layout.v(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(FramePushConst), &framePushConst);

        RTPushConst rtPC{};
        rtPC.frame = static_cast<int>(currentFrame);
        rtPC.lightCount = static_cast<int>(lights.size());

        vkCmdPushConstants(commandBuffer, rtPipeline.layout.v(), VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, sizeof(FramePushConst), sizeof(RTPushConst), &rtPC);

        vkhfp::vkCmdTraceRaysKHR(commandBuffer, &sbt.raygenR, &sbt.missR, &sbt.hitR, &sbt.callR, swap.extent.width, swap.extent.height, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record rt command buffer!");
        }
    }

    void recordAllCommandBuffers() { // record every command buffer
        if (rtEnabled) {
            recordRTCommandBuffers();
        }
        else {
            recordDeferredCommandBuffers();
            recordShadowCommandBuffers();
            recordLightingCommandBuffers();
            recordWBOITCommandBuffers();
        }

        recordCompCommandBuffers();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void recreateSwap() {
        std::cout << "Recreating swap chain...\n";
        int width = 0, height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);
        vkDeviceWaitIdle(device); // wait for the device to be idle

        swap.swapChain.reset(); // reset the SC

        createSC();
        createSCImageViews();
        setupTextures(false);

        // update the descriptorsets
        updateDescriptorSets();

        // create the pipelines
        setupPipelines(false);

        // create the framebuffers
        createFrameBuffers(false);

        initializeMouseInput(true);
    }

    void drawFrame() {
        currentFrame = (currentFrame + 1) % maxFrames;
        framePushConst.frame = currentFrame;

        vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, inFlightFences[currentFrame].p());

        // acquire the next image from the swapchain
        VkResult result = vkAcquireNextImageKHR(device, swap.swapChain.v(), UINT64_MAX, imageAvailableSemaphores[currentFrame].v(), VK_NULL_HANDLE, &swap.imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            vkDeviceWaitIdle(device);
            recreateSwap();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        recordAllCommandBuffers();

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        std::vector<VkSubmitInfo> submitInfos;
        std::vector<VkCommandBuffer> shadowCmds;

        if (!rtEnabled) {
            shadowCmds.reserve(lights.size());
            for (size_t i = 0; i < lights.size(); i++) {
                size_t index = (i * maxFrames) + currentFrame;
                shadowCmds.push_back(shadowMapCommandBuffers.primary.buffers[index].v());
            }

            submitInfos.push_back(vkh::createSubmitInfo(deferredCommandBuffers.primary[currentFrame].p(), 1, &waitStage, imageAvailableSemaphores[currentFrame], deferredSemaphores[currentFrame]));
            submitInfos.push_back(vkh::createSubmitInfo(shadowCmds.data(), shadowCmds.size(), &waitStage, deferredSemaphores[currentFrame], shadowSemaphores[currentFrame]));
            submitInfos.push_back(vkh::createSubmitInfo(lightingPassCommandBuffers.primary[currentFrame].p(), 1, &waitStage, shadowSemaphores[currentFrame], wboitSemaphores[currentFrame]));
            submitInfos.push_back(vkh::createSubmitInfo(wboitCommandBuffers.primary[currentFrame].p(), 1, &waitStage, wboitSemaphores[currentFrame], compSemaphores[currentFrame]));
            submitInfos.push_back(vkh::createSubmitInfo(compCommandBuffers.primary[currentFrame].p(), 1, &waitStage, compSemaphores[currentFrame], renderFinishedSemaphores[currentFrame]));
        }
        else {
            submitInfos.push_back(vkh::createSubmitInfo(rtCommandBuffers.primary[currentFrame].p(), 1, &waitStage, imageAvailableSemaphores[currentFrame], rtSemaphores[currentFrame]));
            submitInfos.push_back(vkh::createSubmitInfo(compCommandBuffers.primary[currentFrame].p(), 1, &waitStage, rtSemaphores[currentFrame], renderFinishedSemaphores[currentFrame]));
        }

        // submit all command buffers in a single call
        if (vkQueueSubmit(graphicsQueue, static_cast<uint32_t>(submitInfos.size()), submitInfos.data(), inFlightFences[currentFrame].v()) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit command buffers!");
        }

        // present the image
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = renderFinishedSemaphores[currentFrame].p();
        presentInfo.pSwapchains = swap.swapChain.p();
        presentInfo.swapchainCount = 1;
        presentInfo.pImageIndices = &swap.imageIndex;
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        //check if the swap chain is out of date (window was resized, etc):
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(device);
            recreateSwap();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
    }

    void calcFps() {
        static auto lastUpdated = utils::now();

        // num of frames since fps was last updated
        static size_t frameCount = 0;
        frameCount++;

        // time since fps was last updated
        double elapsed = utils::duration<milliseconds>(lastUpdated).count() / 1000.0;

        // calculate FPS every 200ms
        if (elapsed >= 0.2) {
            fps = static_cast<uint32_t>(frameCount / elapsed);
            frameCount = 0;
            lastUpdated = utils::now();
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            calcFps();
            glfwPollEvents();
            handleKeyboardInput();
            updateUBO();
            if (rtEnabled) updateTLAS();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
        imguiCleanup();
    }

    void initVulkan() {
        // initialize Vulkan components
        auto now = utils::now();
        validateFiles();
        createInstance();
        createSurface();
        pickDevice();
        createLogicalDevice();
        initQueues();
        createSC();
        setupFences();
        createSemaphores();

        commandPool = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        initializeMouseInput(true);
        loadScene();

        // create buffers and textures
        createModelBuffers(false);

        if (rtEnabled) {
            setupAccelerationStructures();
        }

        setupTextures(true);
        loadSkybox("night-sky.hdr");
        setupBuffers();
        getAllTextures();

        // setup the descriptorsets and pipelines
        setupDescriptorSets();
        setupPipelines(true);

        if (rtEnabled) {
            createSBT();
        }

        imguiSetup();
        updateUBO();

        // setup the framebuffers and command buffers
        createFrameBuffers(true);
        createCommandBuffers();

        auto duration = utils::duration<milliseconds>(now);
        utils::printDuration(duration);

        std::cout << "Vulkan initialized successfully! Unique models: " << getUniqueModels() << "\n";
        utils::sep();
    }
};

int main() {
    {
        Engine app;
        try {
            app.run();
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    utils::sep();
    utils::sep();
    std::cout << "CLOSING ENGINE\n";
    utils::sep();
    utils::sep();

    return EXIT_SUCCESS;
}
