#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(push_constant) uniform PushConstant {
    uint texture_id;
} push;

layout (location = 0) in vec3 in_uvw;
layout (location = 0) out vec4 out_frag_color;

layout (set = 0, binding = 0) uniform samplerCube sampler_cubemaps[];

void main() 
{
	out_frag_color = texture(sampler_cubemaps[nonuniformEXT(push.texture_id)], in_uvw);
}
