import pip
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

def check_requirements():
    # only check for Kornia, autograd and torch, as these should be the important ones, the rest might be on different versions
    dependencies = [
        'autograd==1.7.0',
        'colorama==0.4.6',
        'contourpy==1.3.1',
        'cycler==0.12.1',
        'fake-bpy-module-latest==20250205',
        'filelock==3.18.0',
        'fonttools==4.56.0',
        'fsspec==2025.3.2',
        'Jinja2==3.1.6',
        'kiwisolver==1.4.8',
        'kornia==0.8.1',
        # 'kornia_rs==0.1.9',
        'MarkupSafe==3.0.2',
        'matplotlib==3.10.0',
        'mpmath==1.3.0',
        'networkx==3.4.2',
        'numpy==2.2.2',
        'opencv-contrib-python==4.11.0.86',
        'opencv-python==4.11.0.86',
        'packaging==24.2',
        'pandas==2.3.1',
        'pillow==11.1.0',
        'pyparsing==3.2.1',
        'pyquaternion==0.9.9',
        'python-dateutil==2.9.0.post0',
        'pytz==2025.2',
        'scipy==1.15.1',
        'six==1.17.0',
        'sympy==1.14.0',
        # 'torch==2.7.0+cu126',
        # 'torchaudio==2.7.0+cu126',
        # 'torchvision==0.22.0+cu126',
        'torch==2.7.0',
        'torchaudio==2.7.0',
        'torchvision==0.22.0',
        'tqdm==4.67.1',
        'typing_extensions==4.13.2',
        'tzdata==2025.2',
    ]


    try:
        pkg_resources.require(dependencies)
        print("All dependencies are satisfied.")
    except (DistributionNotFound, VersionConflict) as e:
        for dependency in dependencies:
            print("Trying to install:", dependency)
            try:
                pip.main(['install', dependency])
            except Exception as install_error:
                print(f"Failed to install {dependency}: {install_error}")

if __name__ == "__main__":
    check_requirements()