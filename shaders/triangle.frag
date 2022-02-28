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
const float attenuation = 30;
const vec3 lightColor = vec3(0.5, 0.7, 0.8);

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

vec3 sampleId(uint id)
{
    return texture(nonuniformEXT(sampler2D(textures[id], immutable_sampler)), TexCoords).rgb;
}

// Sources:
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
mat3 ACESInputMat = mat3
(
    vec3(0.59719, 0.35458, 0.04823),
    vec3(0.07600, 0.90834, 0.01566),
    vec3(0.02840, 0.13383, 0.83777)
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
mat3 ACESOutputMat = mat3
(
    vec3(1.60475, -0.53108, -0.07367),
    vec3(-0.10208, 1.10813, -0.00605),
    vec3(-0.00327, -0.07276, 1.07602)
);

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 ACESFitted(vec3 color)
{
    color = color * ACESInputMat;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = color * ACESOutputMat;

    // Clamp to [0, 1]
    color = clamp(color, 0.0, 1.0);

    return color;
}

void main() {
    /* const vec3 albedo = vec3(0.5,0.5,0.5); */
    const vec3 albedo = sampleId(push.texture_id);

    const vec3 mrao = sampleId(push.met_rogh_id);
    const vec3 normal_color = sampleId(push.normal_id);

    vec3 N = normalize(TBN * (normal_color * 2.0 - 1.0));
    vec3 V = normalize(ubo.camPos - WorldPos);

    vec3 Lo = vec3(0.0);
    vec3 L = normalize(lightPos() - WorldPos);
    vec3 H = normalize(V + L);
    vec3 radiance     = lightColor * attenuation; 

    vec3 F0 = vec3(0.04);
    F0      = mix(F0, albedo, mrao.r);
    vec3 F  = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float NDF = DistributionGGX(N, H, mrao.g);       
    float G   = GeometrySmith(N, V, L, mrao.g);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0)  + 0.0001;
    vec3 specular     = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - mrao.r;

    float NdotL = max(dot(N, L), 0.0);        
    Lo += (kD * albedo / PI + specular) * radiance * NdotL;

    /* const float ao = sampleId(push.ao_id).r; */
    vec3 ambient = vec3(0.5) * albedo * mrao.b;
    vec3 color   = ambient + Lo;
    color = ACESFitted(color);
    FragColor = vec4(color, 1.0);
}

