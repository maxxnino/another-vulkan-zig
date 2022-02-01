#version 450

layout(location = 0) in vec3 in_position;
layout (location = 0) out vec3 out_uvw;

layout(set = 2, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() 
{
    out_uvw = in_position;
    vec4 pos = ubo.proj * ubo.view * vec4(in_position.xyz, 0);

    // make sure that the depth after w divide will be 1.0 (so that the z-buffering will work)
    pos.z = pos.w;
	gl_Position = pos;
}
