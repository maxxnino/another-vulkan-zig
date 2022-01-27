#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
} ubo;

layout(push_constant) uniform PushConsts {
    mat4 view;
    mat4 proj;
} push;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = push.proj * push.view * ubo.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
}
