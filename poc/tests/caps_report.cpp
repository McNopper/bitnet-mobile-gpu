// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Capability report: the human-readable console dump and the Markdown
// capability snapshot written by --caps-out.

#include "backend/extension_families.hpp"
#include "tests/suites.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

void print_extended_caps(const bitnet::DeviceCaps& caps) {
    std::printf("[caps] storage 8-bit=%s 16-bit=%s; subgroup control=%s full=%s range=%u..%u "
                "max_groups=%u required_compute=%s\n",
                caps.storage_8bit ? "true" : "false", caps.storage_16bit ? "true" : "false",
                caps.subgroup_size_control ? "true" : "false",
                caps.compute_full_subgroups ? "true" : "false", caps.min_subgroup_size,
                caps.max_subgroup_size, caps.max_compute_workgroup_subgroups,
                caps.required_subgroup_size_compute ? "true" : "false");
    std::printf("[caps] ARM core props=%s pixel/texel/fma=%u/%u/%u\n",
                caps.arm_shader_core_properties ? "present" : "absent", caps.arm_pixel_rate,
                caps.arm_texel_rate, caps.arm_fma_rate);
    std::printf("[caps] coopmat ext=%s s8*s8->s32_match=%s shapes=%u m=%u n=%u k=%u scope=%u "
                "stages=0x%x\n",
                caps.cooperative_matrix ? "present" : "absent",
                caps.coop_mat_shape_s8_s32 ? "true" : "false", caps.coop_mat_shape_count,
                caps.coop_mat_msize, caps.coop_mat_nsize, caps.coop_mat_ksize, caps.coop_mat_scope,
                caps.coop_mat_supported_stages);
    std::printf("[caps] QCOM image processing=%s weighted=%s box=%s block_match=%s "
                "max_phases=%u weight=%ux%u block=%ux%u box_size=%ux%u\n",
                caps.qcom_image_processing ? "present" : "absent",
                caps.qcom_texture_sample_weighted ? "true" : "false",
                caps.qcom_texture_box_filter ? "true" : "false",
                caps.qcom_texture_block_match ? "true" : "false",
                caps.qcom_max_weight_filter_phases, caps.qcom_max_weight_filter_width,
                caps.qcom_max_weight_filter_height, caps.qcom_max_block_match_width,
                caps.qcom_max_block_match_height, caps.qcom_max_box_filter_width,
                caps.qcom_max_box_filter_height);
}

void print_relevant_extensions(const std::vector<std::string>& extensions) {
    const auto print_family = [&extensions](const char* family, size_t begin, size_t end) {
        uint32_t present = 0;
        for (size_t i = begin; i < end; ++i) {
            if (bitnet::extension_present(extensions, bitnet::kRelevantExtensionNames[i])) {
                ++present;
            }
        }
        std::printf("[caps] extension family %-7s relevant present %u/%zu\n", family, present,
                    end - begin);
    };
    print_family("KHR/EXT", 0, bitnet::kKhrExtFamilyCount);
    print_family("ARM", bitnet::kArmFamilyBegin, bitnet::kArmFamilyBegin + bitnet::kArmFamilyCount);
    print_family("QCOM", bitnet::kQcomFamilyBegin,
                 bitnet::kQcomFamilyBegin + bitnet::kQcomFamilyCount);
    print_family("NV", bitnet::kNvFamilyBegin, bitnet::kNvFamilyBegin + bitnet::kNvFamilyCount);

    for (const std::string_view name : bitnet::kRelevantExtensionNames) {
        std::printf("[caps] extension %-45s %s\n", std::string(name).c_str(),
                    bitnet::extension_present(extensions, name) ? "present" : "absent");
    }
}

// Markdown fragment with the full capability snapshot (the file
// of --caps-out).
void write_caps(const char* path, const bitnet::DeviceCaps& caps,
                const std::vector<std::string>& extensions) {
    std::ofstream f(path);
    if (!f) {
        std::printf("caps-out: cannot write %s\n", path);
        return;
    }
    f << "# Vulkan device capabilities\n\n";
    f << "| Property | Value |\n| --- | --- |\n";
    f << "| device_name | " << caps.device_name << " |\n";
    f << "| driver_version | " << caps.driver_version << " |\n";
    f << "| api_version | " << bitnet::version_major(caps.api_version) << '.'
      << bitnet::version_minor(caps.api_version) << '.' << bitnet::version_patch(caps.api_version)
      << " |\n";
    f << "| max_shared_memory_bytes | " << caps.max_shared_memory_bytes << " |\n";
    f << "| max_workgroup_invocations | " << caps.max_workgroup_invocations << " |\n";
    f << "| subgroup_size | " << caps.subgroup_size << " |\n";
    f << "| subgroup_arithmetic | " << (caps.subgroup_arithmetic ? "true" : "false") << " |\n";
    f << "| shader_integer_dot_product | " << (caps.shader_integer_dot_product ? "true" : "false")
      << " |\n";
    f << "| dot_4x8_packed_signed_accelerated | "
      << (caps.dot_4x8_packed_signed_accelerated ? "true" : "false") << " |\n";
    f << "| dot_8bit_signed_accelerated | " << (caps.dot_8bit_signed_accelerated ? "true" : "false")
      << " |\n";
    f << "| cooperative_matrix | " << (caps.cooperative_matrix ? "true" : "false") << " |\n";
    f << "| cooperative_vector_nv | " << (caps.cooperative_vector_nv ? "true" : "false") << " |\n";
    f << "| coop_mat_shape_s8_s32 | " << (caps.coop_mat_shape_s8_s32 ? "true" : "false") << " |\n";
    f << "| coop_mat_shape_count | " << caps.coop_mat_shape_count << " |\n";
    f << "| coop_mat_shape_mnk_scope | " << caps.coop_mat_msize << 'x' << caps.coop_mat_nsize << 'x'
      << caps.coop_mat_ksize << " scope=" << caps.coop_mat_scope << " |\n";
    f << "| coop_mat_supported_stages | " << caps.coop_mat_supported_stages << " |\n";
    f << "| storage_8bit | " << (caps.storage_8bit ? "true" : "false") << " |\n";
    f << "| storage_16bit | " << (caps.storage_16bit ? "true" : "false") << " |\n";
    f << "| subgroup_size_control | " << (caps.subgroup_size_control ? "true" : "false") << " |\n";
    f << "| compute_full_subgroups | " << (caps.compute_full_subgroups ? "true" : "false")
      << " |\n";
    f << "| min_subgroup_size | " << caps.min_subgroup_size << " |\n";
    f << "| max_subgroup_size | " << caps.max_subgroup_size << " |\n";
    f << "| max_compute_workgroup_subgroups | " << caps.max_compute_workgroup_subgroups << " |\n";
    f << "| required_subgroup_size_compute | "
      << (caps.required_subgroup_size_compute ? "true" : "false") << " |\n";
    f << "| arm_shader_core_properties | " << (caps.arm_shader_core_properties ? "true" : "false")
      << " |\n";
    f << "| arm_pixel_rate | " << caps.arm_pixel_rate << " |\n";
    f << "| arm_texel_rate | " << caps.arm_texel_rate << " |\n";
    f << "| arm_fma_rate | " << caps.arm_fma_rate << " |\n";
    f << "| qcom_image_processing | " << (caps.qcom_image_processing ? "true" : "false") << " |\n";
    f << "| qcom_texture_sample_weighted | "
      << (caps.qcom_texture_sample_weighted ? "true" : "false") << " |\n";
    f << "| qcom_texture_box_filter | " << (caps.qcom_texture_box_filter ? "true" : "false")
      << " |\n";
    f << "| qcom_texture_block_match | " << (caps.qcom_texture_block_match ? "true" : "false")
      << " |\n";
    f << "| qcom_max_weight_filter_phases | " << caps.qcom_max_weight_filter_phases << " |\n";
    f << "| qcom_max_weight_filter_dimension | " << caps.qcom_max_weight_filter_width << 'x'
      << caps.qcom_max_weight_filter_height << " |\n";
    f << "| qcom_max_block_match_region | " << caps.qcom_max_block_match_width << 'x'
      << caps.qcom_max_block_match_height << " |\n";
    f << "| qcom_max_box_filter_block_size | " << caps.qcom_max_box_filter_width << 'x'
      << caps.qcom_max_box_filter_height << " |\n";
    f << "| max_workgroup_size | " << caps.max_workgroup_size[0] << '/'
      << caps.max_workgroup_size[1] << '/' << caps.max_workgroup_size[2] << " |\n";
    f << "| max_workgroup_count | " << caps.max_workgroup_count[0] << '/'
      << caps.max_workgroup_count[1] << '/' << caps.max_workgroup_count[2] << " |\n";
    f << "| max_storage_buffer_range | " << caps.max_storage_buffer_range << " |\n";
    f << "| max_memory_allocation_size | " << caps.max_memory_allocation_size << " |\n";
    f << "\n## Relevant extension presence\n\n";
    f << "| Extension | Value |\n| --- | --- |\n";
    for (const std::string_view name : bitnet::kRelevantExtensionNames) {
        f << "| " << name << " | "
          << (bitnet::extension_present(extensions, name) ? "present" : "absent") << " |\n";
    }
    f << "\n## Device extensions (" << extensions.size() << ")\n\n";
    for (const std::string& ext : extensions) {
        f << "- " << ext << "\n";
    }
    std::printf("caps written to %s\n", path);
}