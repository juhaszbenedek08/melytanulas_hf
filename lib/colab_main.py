import main
import sys
import path_util

if __name__ == '__main__':
    path_util.data_dir.symlink_to('drive/MyDrive/melytanulas_hf/data')
    path_util.out_dir.symlink_to('drive/MyDrive/melytanulas_hf/out')

    if len(sys.argv) == 1:
        pass
        # TODO continue last training

    main.main()
