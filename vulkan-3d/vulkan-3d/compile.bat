@@echo off
glslangValidator -V engine.vert -o vertex_shader.spv > GLSLerr.txt
glslangValidator -V engine.frag -o fragment_shader.spv >> GLSLerr.txt

glslangValidator -V shadow.vert -o shadow_vert_shader.spv
glslangValidator -V shadow.frag -o shadow_frag_shader.spv
