#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform sampler immutable_sampler;

layout(push_constant) uniform PushConsts {
    uint texture_id;
} push;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_frag_color;


void main() {
    out_frag_color = texture(nonuniformEXT(sampler2D(textures[push.texture_id], immutable_sampler)), in_uv);
}
