@@echo off
glslangValidator -V engine.vert -o compiled/vertex_shader.spv > shaderlogs/GLSLerr.txt
glslangValidator -V engine.frag -o compiled/fragment_shader.spv >> shaderlogs/GLSLerr.txt

glslangValidator -V shadow.vert -o compiled/shadow_vert_shader.spv > shaderlogs/shaderErr.txt
glslangValidator -V shadow.frag -o compiled/shadow_frag_shader.spv >> shaderlogs/shaderErr.txt

glslangValidator -V sky.vert -o compiled/sky_vert_shader.spv > shaderlogs/skyErr.txt
glslangValidator -V sky.frag -o compiled/sky_frag_shader.spv >> shaderlogs/skyErr.txt
