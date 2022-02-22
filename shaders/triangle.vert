#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_tex_coord;
layout(location = 2) in vec4 in_normal;
layout(location = 3) in vec4 in_tangent;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out mat3 out_tbn;

layout(set = 2, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camPos;
} ubo;

void main() {
    mat4 model_view = ubo.view * ubo.model;
    vec4 pos = vec4(in_position.xyz, 1.0);
    gl_Position = ubo.proj * model_view * pos;

    out_world_pos = vec3(model_view * pos);
    out_uv = in_tex_coord;

    mat3 model = mat3(ubo.model);
    vec3 N     = normalize(model * in_normal.xyz);
    vec3 T     = normalize(model * in_tangent.xyz);
    vec3 B     = normalize(model * (cross(in_normal.xyz, in_tangent.xyz)) * in_tangent.w);
    out_tbn    = mat3(T, B, N);

}
