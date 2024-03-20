import subprocess

def main():
    command = f"cd .. && cd CAT-Net-main && python tools/infer.py"

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()