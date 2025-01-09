import os
import sys
import argparse
import pgpy
from pathlib import Path
import shutil

def get_fingerprint(key_path):
    """
    从给定的密钥文件中提取指纹。
    
    :param key_path: 密钥文件的路径
    :return: 指纹字符串
    """
    with open(key_path, 'rb') as f:
        key_data = f.read()
    try:
        key, _ = pgpy.PGPKey.from_blob(key_data)
        return key.fingerprint
    except Exception as e:
        raise ValueError(f"无法解析密钥文件 {key_path.name}: {e}")

def get_keyid(fingerprint):
    """
    从指纹中提取 KeyID（最后 16 个字符）。
    
    :param fingerprint: 指纹字符串
    :return: KeyID 字符串
    """
    return fingerprint[-16:]

def count_consecutive_end_chars(keyid):
    """
    计算 KeyID 末尾连续相同字符的数量。
    
    :param keyid: KeyID 字符串
    :return: 连续相同字符的数量（整数）
    """
    if not keyid:
        return 0
    last_char = keyid[-1]
    count = 1
    for char in reversed(keyid[:-1]):
        if char == last_char:
            count += 1
        else:
            break
    return count

def sanitize_filename(filename):
    """
    对文件名进行简单的清理，避免非法字符。
    
    :param filename: 原始文件名
    :return: 清理后的文件名
    """
    return "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()

def categorize_keys(input_dir, output_dir):
    """
    分类 GPG 私钥文件并将其移动到输出目录。
    
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"输入目录 {input_dir} 不存在或不是一个目录。")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # 遍历输入目录下所有 .gpg 文件
    for key_file in input_path.glob('*.gpg'):
        try:
            fingerprint = get_fingerprint(key_file)
            keyid = get_keyid(fingerprint)
            consecutive_count = count_consecutive_end_chars(keyid)

            # 创建对应的分类子文件夹
            category_dir = output_path / str(consecutive_count)
            category_dir.mkdir(parents=True, exist_ok=True)

            # 获取原文件名（不包含路径）
            original_filename = key_file.stem  # 不包含扩展名
            sanitized_filename = sanitize_filename(original_filename)

            # 构建新的文件名为 <KeyID>-<原文件名>.gpg
            new_filename = f"{keyid}-{sanitized_filename}.gpg"
            destination = category_dir / new_filename

            # 移动文件到目标位置
            shutil.move(str(key_file), str(destination))

            print(f"已处理 KeyID: {keyid}, 分类 {consecutive_count}, 移动为 {new_filename}")
        except Exception as e:
            print(f"处理文件 {key_file.name} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='根据 KeyID 末尾连续字符数分类 GPG 私钥文件并重命名。')
    parser.add_argument('input_dir', type=str, help='包含 GPG 私钥文件的指定目录。')
    parser.add_argument('output_dir', type=str, help='分类后文件的输出目录。')
    args = parser.parse_args()

    categorize_keys(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()

