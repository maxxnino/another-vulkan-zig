#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform sampler immutable_sampler;

layout(set = 2, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camPos;
} ubo;

layout(push_constant) uniform PushConsts {
    uint texture_id;
    uint normal_id;
    uint met_rogh_id;
    uint ao_id;
    float x;
    float y;
    float z;
} push;

const float PI = 3.14159265359;
vec3 lightPos() {
    return vec3(push.x, push.y, push.z);
}

layout (location = 0) in vec3 WorldPos;
layout (location = 1) in vec2 TexCoords;
layout (location = 2) in mat3 TBN;

layout(location = 0) out vec4 FragColor;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}
const float attenuation = 1;
const vec3 lightColor = vec3(1, 1, 1);
const float gamma = 2.2;

vec3 gammaCorrectTexture(uint id)
{
    const vec3 base_color = texture(nonuniformEXT(sampler2D(textures[id], immutable_sampler)), TexCoords).rgb;
    return pow(base_color.rgb, vec3(gamma));
}

vec3 sampleId(uint id)
{
    return texture(nonuniformEXT(sampler2D(textures[id], immutable_sampler)), TexCoords).rgb;
}

void main() {
    /* const vec3 albedo = vec3(0.5,0.5,0.5); */
    const vec3 albedo = sampleId(push.texture_id);

    const vec3 mr = sampleId(push.met_rogh_id);
    const vec3 normal_color = sampleId(push.normal_id);

    vec3 N = normalize(TBN * (normal_color * 2.0 - 1.0));
    vec3 V = normalize(ubo.camPos - WorldPos);

    vec3 Lo = vec3(0.0);
    vec3 L = normalize(lightPos() - WorldPos);
    vec3 H = normalize(V + L);
    vec3 radiance     = lightColor * attenuation; 

    vec3 F0 = vec3(0.04);
    F0      = mix(F0, albedo, mr.r);
    vec3 F  = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float NDF = DistributionGGX(N, H, mr.g);       
    float G   = GeometrySmith(N, V, L, mr.g);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0)  + 0.0001;
    vec3 specular     = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - mr.r;

    float NdotL = max(dot(N, L), 0.0);        
    Lo += (kD * albedo / PI + specular) * radiance * NdotL;

    const float ao = sampleId(push.ao_id).r;
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color   = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}

