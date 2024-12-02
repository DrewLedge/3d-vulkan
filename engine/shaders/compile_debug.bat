@echo off

if exist compiled rmdir /s /q compiled
if exist shaderlogs rmdir /s /q shaderlogs
mkdir compiled
mkdir shaderlogs

glslc --target-env=vulkan1.3 deferred.vert -I ./includes/ -o compiled/deferred_vertex_shader.spv > shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 deferred.frag -I ./includes/ -o compiled/deferred_fragment_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 engine.vert -I ./includes/ -o compiled/vertex_shader.spv > shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 engine.frag -I ./includes/ -o compiled/fragment_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 shadow.vert -I ./includes/ -o compiled/shadow_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 shadow.frag -o compiled/shadow_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 sky.vert -o compiled/sky_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 sky.frag -o compiled/sky_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 wboit.vert -I ./includes/ -o compiled/wboit_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 wboit.frag -I ./includes/ -o compiled/wboit_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 composition.vert -o compiled/composition_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 composition.frag -o compiled/composition_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1


glslc --target-env=vulkan1.3 pathtracing/closehit.rchit -o compiled/closehit_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 pathtracing/shadowhit.rchit -o compiled/shadowhit_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 pathtracing/gen.rgen -o compiled/gen_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 pathtracing/miss.rmiss -o compiled/miss_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 pathtracing/shadowmiss.rmiss -o compiled/shadowmiss_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 pathtracing/present.vert -o compiled/rt_present_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 pathtracing/present.frag -o compiled/rt_present_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

