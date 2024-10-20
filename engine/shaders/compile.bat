@echo off

if not exist compiled mkdir compiled
if not exist shaderlogs mkdir shaderlogs

glslc --target-env=vulkan1.3 -O engine.vert -I ./includes/ -o compiled/vertex_shader.spv > shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O engine.frag -I ./includes/ -o compiled/fragment_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 -O shadow.vert -I ./includes/ -o compiled/shadow_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O shadow.frag -o compiled/shadow_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 -O sky.vert -o compiled/sky_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O sky.frag -o compiled/sky_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 -O wboit.vert -I ./includes/ -o compiled/wboit_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O wboit.frag -I ./includes/ -o compiled/wboit_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc --target-env=vulkan1.3 -O composition.vert -o compiled/composition_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O composition.frag -o compiled/composition_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1


glslc --target-env=vulkan1.3 -O pathtracing/anyhit.rahit -o compiled/anyhit_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O pathtracing/closehit.rchit -o compiled/closehit_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O pathtracing/gen.rgen -o compiled/gen_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O pathtracing/intersection.rint -o compiled/intersection_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc --target-env=vulkan1.3 -O pathtracing/miss.rmiss -o compiled/miss_shader.spv >> shaderlogs/ShaderError.txt 2>&1


