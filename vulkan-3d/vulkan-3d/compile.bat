@@echo off
glslangValidator -V engine.vert -o vertex_shader.spv > GLSLerr.txt
glslangValidator -V engine.frag -o fragment_shader.spv >> GLSLerr.txt


