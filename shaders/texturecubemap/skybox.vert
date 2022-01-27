#version 450

layout (location = 0) in vec3 inPos;

layout (binding = 0) uniform UBO 
{
	mat4 model;
} ubo;

layout(push_constant) uniform PushConsts {
    mat4 view;
    mat4 proj;
} push;

layout (location = 0) out vec3 outUVW;

void main() 
{
    outUVW = inPos;
    vec4 pos = push.proj * push.view * vec4(inPos.xyz, 0);

    // make sure that the depth after w divide will be 1.0 (so that the z-buffering will work)
    pos.z = pos.w;
	gl_Position = pos;
}
