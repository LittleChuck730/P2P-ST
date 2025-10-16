import unmixing_main as ua
import os


def Unmixing_algorithm(input_image_mi_list,
                       input_image_ci_list,
                       window_sizes,
                       fake_wndo_size,
                       alpha_values,
                       output_folder):
    # Batch processing
    for input_image_mi in input_image_mi_list:
        for input_image_ci in input_image_ci_list:
            for in_window_size in window_sizes:
                for alpha in alpha_values:
                    alf = f"{int(alpha * 1000):04d}"
                    sides = f"{in_window_size / fake_wndo_size:.0f}"

                    # Path of the final output image
                    output_image_mi = os.path.join(
                        output_folder,
                        rf'c{sides}&{sides}_α{alf}_'
                        rf'{os.path.splitext(os.path.basename(input_image_mi))[0]}_'
                        rf'{os.path.splitext(os.path.basename(input_image_ci))[0]}.tif'
                    )

                    # Check if the file already exists
                    if os.path.exists(output_image_mi):
                        print(f"File already exists, skipping: {output_image_mi}")
                        continue

                    print(
                        f'==============================---Start processing: MI image={input_image_mi}, '
                        f'Classification image={input_image_ci}, Window size={in_window_size}, α={alpha}'
                        f'---========================================')

                    # 1---Window-based unmixing
                    finally_end = ua.slide_window(input_image_mi,
                                                  input_image_ci,
                                                  in_window_size,
                                                  fake_wndo_size,
                                                  alpha)

                    # 2---Image reconstruction
                    ua.image_restore(input_image_mi,
                                     finally_end,
                                     output_image_mi)

                    print(f'Processing completed, saved to {output_image_mi}')

    print("Batch processing completed!")


if __name__ == "__main__":
    input_image_mi_list = [

        r'C:\Users\Lenovo\Documents\ArcGIS\0x0x0x\重采样回原尺度\Band_1_Resample_res_10.tif',
        r'C:\Users\Lenovo\Documents\ArcGIS\0x0x0x\重采样回原尺度\Band_2_Resample_res_10.tif',
        r'C:\Users\Lenovo\Documents\ArcGIS\0x0x0x\重采样回原尺度\Band_3_Resample_res_10.tif',

    ]

    input_image_ci_list = [
        r'C:\Users\Lenovo\Desktop\分类了.tif',
    ]

    window_sizes = [90]
    alpha_values = [0.1]

    fake_wndo_size = 10

    output_folder = r'C:\Users\Lenovo\Documents\ArcGIS\0x0x0x\end'

    Unmixing_algorithm(input_image_mi_list,
                       input_image_ci_list,
                       window_sizes,
                       fake_wndo_size,
                       alpha_values,
                       output_folder)
