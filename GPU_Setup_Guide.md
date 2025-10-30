# Guide: Setting Up Your NVIDIA GPU (GTX 1650) for Training

This guide explains how to configure your system to use your NVIDIA GeForce GTX 1650 for training the model in this project. Using the GPU will drastically reduce training times compared to the CPU.


**Step 1: Install/Update NVIDIA Drivers**

Ensure you have the latest drivers for your GTX 1650 installed. Outdated drivers are a common cause of problems.

* Visit the official NVIDIA Driver Downloads page.

* Select your GPU model  and your operating system.

* Choose Driver Type: Select the Studio Driver (Recommended for Compute/AI) for better stability, or the Game Ready Driver if you also game frequently.

* Download and install the driver.

* Reboot your computer after installation. This is crucial.

**Step 2: Install PyTorch with CUDA Support**

This is the most important step. You need the version of PyTorch specifically built to work with your NVIDIA GPU's CUDA capabilities.

### (Optional but Recommended) Uninstall existing PyTorch (if you installed a CPU-only version):

``` bash
pip uninstall torch torchvision torchaudio -y
```

- Go to the official PyTorch website: https://pytorch.org/get-started/locally/

- Use the interactive tool: Select the following options:

  - PyTorch Build: Stable

  - Your OS: Windows

  - Package: Pip

  - Language: Python

  - Compute Platform: CUDA (select the latest available version, e.g., CUDA 12.1 or newer. Your GTX 1650 supports these).

  - Copy the generated command. It will look something like this (use the exact command from the website, don't copy this example):
``` bash
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

Run this command in your activated virtual environment's terminal.

**Step 3: Verify the Setup**

After installation, verify that PyTorch can see and use your GPU:

Open your terminal (with your virtual environment activated).

Run this simplified command (avoids f-strings for better terminal compatibility):
``` bash
python -c "import torch; cuda_available = torch.cuda.is_available(); print('CUDA available: ' + str(cuda_available)); device_name = torch.cuda.get_device_name(0) if cuda_available else 'N/A'; print('Device name: ' + device_name)"
``` 

``` bash
Expected Output:

CUDA available: True
Device name: NVIDIA GeForce GTX 1650
```

If it prints True and your GPU name, your setup is correct! The training script will now automatically use your GTX 1650.

If it prints False, double-check your driver installation (Step 1) and ensure you ran the correct PyTorch installation command (Step 2).

**Step 4: Run Your Training**

Now you can run the training step using the full dataset:
``` bash
python main.py train
```

Watch the initial log messages. You should see:
INFO - Using device: cuda

This confirms the GPU is being used. Training will now be significantly faster.

**Step 5: (If Necessary) Address Potential DLL Errors (WinError 1114)**

- If you still encounter an OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed... (this is less likely after correct driver/PyTorch install but can happen on some laptops), follow these steps:

- Open Windows Graphics Settings: Search for "Graphics settings" in the Start Menu.

- Add Python: Click "Browse" under "Custom options for apps" and navigate to your Python executable (e.g., C:\Users\YourUser\AppData\Local\Programs\Python\Python312\python.exe). Add it.

- Set High Performance: Click on Python in the list, select "Options," choose "High performance" (which should list your GTX 1650), and click "Save."

- Restart your terminal/IDE.

By following these steps, your project will leverage the power of your GTX 1650, making the training process much more efficient.