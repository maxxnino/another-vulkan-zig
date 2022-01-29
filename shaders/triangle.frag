#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout (set = 1, binding = 0) uniform sampler2D textures[];

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConsts {
    uint texture_id;
} push;

void main() {
    outColor = texture(textures[nonuniformEXT(push.texture_id)], fragTexCoord);
}
