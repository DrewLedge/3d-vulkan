@echo off

if not exist compiled mkdir compiled
if not exist shaderlogs mkdir shaderlogs

glslc -O engine.vert -I ./includes/ -o compiled/vertex_shader.spv > shaderlogs/ShaderError.txt 2>&1
glslc -O engine.frag -I ./includes/ -o compiled/fragment_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc -O shadow.vert -I ./includes/ -o compiled/shadow_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc -O shadow.frag -o compiled/shadow_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc -O sky.vert -o compiled/sky_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc -O sky.frag -o compiled/sky_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc -O wboit.vert -I ./includes/ -o compiled/wboit_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc -O wboit.frag -I ./includes/ -o compiled/wboit_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc -O composition.vert -o compiled/composition_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc -O composition.frag -o compiled/composition_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1