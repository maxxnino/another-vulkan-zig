#version 450

layout(location = 0) in vec4 in_position;
layout (location = 0) out vec3 out_uvw;

layout(set = 2, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() 
{
    vec4 pos = in_position - vec4(0.5, 0.5, 0.5, 0);
    out_uvw = pos.xyz;
    pos = ubo.proj * ubo.view * pos;

    // make sure that the depth after w divide will be 1.0 (so that the z-buffering will work)
	gl_Position = pos.xyww;

}
