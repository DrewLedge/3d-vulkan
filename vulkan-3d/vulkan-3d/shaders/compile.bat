@echo off

if not exist compiled mkdir compiled
if not exist shaderlogs mkdir shaderlogs

glslc engine.vert -o compiled/vertex_shader.spv > shaderlogs/ShaderError.txt 2>&1
glslc engine.frag -o compiled/fragment_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc shadow.vert -o compiled/shadow_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc shadow.frag -o compiled/shadow_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc sky.vert -o compiled/sky_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc sky.frag -o compiled/sky_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc peel.vert -o compiled/peel_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc peel.frag -o compiled/peel_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1

glslc composition.vert -o compiled/composition_vert_shader.spv >> shaderlogs/ShaderError.txt 2>&1
glslc composition.frag -o compiled/composition_frag_shader.spv >> shaderlogs/ShaderError.txt 2>&1