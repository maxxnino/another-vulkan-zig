#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coord;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec4 in_tangent;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec4 out_tangent;

layout(set = 2, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camPos;
} ubo;

void main() {
    mat4 model_view = ubo.view * ubo.model;
    vec4 pos = vec4(in_position, 1.0);
    gl_Position = ubo.proj * model_view * pos;

    /* out_normal = mat3(ubo.model) * in_normal; */
    /* out_world_pos = vec3(0,0,0); */
    /* out_uv = in_tex_coord; */
    out_world_pos = vec3(model_view * pos);
    out_normal = in_normal * mat3(model_view);
    out_uv = in_tex_coord;
    out_tangent = in_tangent;
}
