import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
from pathlib import Path
import queue
import time

class FP8ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FP8 Model Converter")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        self.process = None
        self.is_running = False
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.t5xxl_var = tk.BooleanVar()
        self.keep_distillation_var = tk.BooleanVar()
        self.calib_samples_var = tk.IntVar(value=3072)
        self.num_iter_var = tk.IntVar(value=500)
        
        self.setup_ui()
        self.check_output_queue()
        
    def setup_ui(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="FP8 Model Converter with Learned Rounding", 
                               font=("TkDefaultFont", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Input file
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Entry(file_frame, textvariable=self.input_path, width=60).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, pady=(0, 5))
        
        # Output file
        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Entry(file_frame, textvariable=self.output_path, width=60).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_output_file).grid(row=1, column=2, pady=(0, 5))
        
        # Auto-generate output filename button
        ttk.Button(file_frame, text="Auto-generate Output", command=self.auto_generate_output).grid(row=2, column=1, pady=(5, 0))
        
        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Conversion Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        
        # Model type options
        ttk.Checkbutton(options_frame, text="T5XXL Model (exclude certain layers)", 
                       variable=self.t5xxl_var).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Checkbutton(options_frame, text="Keep Distillation Layers", 
                       variable=self.keep_distillation_var).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # Parameters section
        params_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        
        # Calibration samples
        ttk.Label(params_frame, text="Calibration Samples:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        calib_frame = ttk.Frame(params_frame)
        calib_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        ttk.Scale(calib_frame, from_=512, to=8192, variable=self.calib_samples_var, 
                 orient=tk.HORIZONTAL, length=200, command=self.update_calib_label).pack(side=tk.LEFT)
        self.calib_label = ttk.Label(calib_frame, text=str(self.calib_samples_var.get()))
        self.calib_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Number of iterations
        ttk.Label(params_frame, text="Optimization Iterations:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        iter_frame = ttk.Frame(params_frame)
        iter_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        ttk.Scale(iter_frame, from_=100, to=2000, variable=self.num_iter_var, 
                 orient=tk.HORIZONTAL, length=200, command=self.update_iter_label).pack(side=tk.LEFT)
        self.iter_label = ttk.Label(iter_frame, text=str(self.num_iter_var.get()))
        self.iter_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        
        self.convert_button = ttk.Button(button_frame, text="Start Conversion", 
                                       command=self.start_conversion, style="Accent.TButton")
        self.convert_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Conversion", 
                                    command=self.stop_conversion, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="Clear Log", 
                                     command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 5))
        
        # Status label
        self.status_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.status_label.grid(row=6, column=0, columnspan=3)
        
        # Output log
        log_frame = ttk.LabelFrame(main_frame, text="Conversion Log", padding="10")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80, 
                                                state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind input change to auto-update output
        self.input_path.trace('w', self.on_input_change)
    
    def update_calib_label(self, value):
        self.calib_label.config(text=str(int(float(value))))
    
    def update_iter_label(self, value):
        self.iter_label.config(text=str(int(float(value))))
    
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select input safetensors file",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
    
    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save output file as",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")],
            defaultextension=".safetensors"
        )
        if filename:
            self.output_path.set(filename)
    
    def on_input_change(self, *args):
        # Auto-update output path when input changes
        if self.input_path.get() and not self.output_path.get():
            self.auto_generate_output()
    
    def auto_generate_output(self):
        input_file = self.input_path.get()
        if not input_file:
            return
        
        base_name = os.path.splitext(input_file)[0]
        distill_str = "_nodistill" if self.keep_distillation_var.get() else ""
        output_file = f"{base_name}_float8_e4m3fn_scaled_learned{distill_str}_svd.safetensors"
        self.output_path.set(output_file)
    
    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def validate_inputs(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file.")
            return False
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return False
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file.")
            return False
        
        # Check if trying to overwrite input
        if os.path.abspath(self.input_path.get()) == os.path.abspath(self.output_path.get()):
            messagebox.showerror("Error", "Output file cannot be the same as input file.")
            return False
        
        return True
    
    def start_conversion(self):
        if not self.validate_inputs():
            return
        
        if self.is_running:
            messagebox.showwarning("Warning", "Conversion is already running!")
            return
        
        # Clear log and start
        self.clear_log()
        self.is_running = True
        self.convert_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress.start()
        self.progress_var.set("Converting...")
        
        # Start conversion in separate thread
        self.conversion_thread = threading.Thread(target=self.run_conversion, daemon=True)
        self.conversion_thread.start()
    
    def run_conversion(self):
        try:
            # Find the original script
            script_path = "convert_fp8_scaled_learned_svd_fast.py"
            if not os.path.exists(script_path):
                self.output_queue.put(("ERROR", f"Cannot find script: {script_path}"))
                return
            
            # Build command
            cmd = [
                sys.executable, script_path,
                "--input", self.input_path.get(),
                "--output", self.output_path.get(),
                "--calib_samples", str(self.calib_samples_var.get()),
                "--num_iter", str(self.num_iter_var.get())
            ]
            
            if self.t5xxl_var.get():
                cmd.append("--t5xxl")
            
            if self.keep_distillation_var.get():
                cmd.append("--keep_distillation")
            
            self.output_queue.put(("LOG", f"Running command: {' '.join(cmd)}"))
            
            # Run the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line
            while True:
                if self.process.poll() is not None:
                    break
                
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(("LOG", line.rstrip()))
            
            # Get any remaining output
            remaining_output = self.process.stdout.read()
            if remaining_output:
                for line in remaining_output.splitlines():
                    if line.strip():
                        self.output_queue.put(("LOG", line))
            
            # Check return code
            return_code = self.process.returncode
            if return_code == 0:
                self.output_queue.put(("SUCCESS", "Conversion completed successfully!"))
            else:
                self.output_queue.put(("ERROR", f"Conversion failed with return code: {return_code}"))
                
        except Exception as e:
            self.output_queue.put(("ERROR", f"Error during conversion: {str(e)}"))
        finally:
            self.output_queue.put(("DONE", ""))
    
    def stop_conversion(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.log_message("Conversion stopped by user.")
            except:
                try:
                    self.process.kill()
                    self.log_message("Conversion forcefully stopped.")
                except:
                    pass
        
        self.is_running = False
        self.convert_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress.stop()
        self.progress_var.set("Stopped")
    
    def check_output_queue(self):
        try:
            while True:
                msg_type, message = self.output_queue.get_nowait()
                
                if msg_type == "LOG":
                    self.log_message(message)
                elif msg_type == "ERROR":
                    self.log_message(f"ERROR: {message}")
                    messagebox.showerror("Conversion Error", message)
                elif msg_type == "SUCCESS":
                    self.log_message(message)
                    messagebox.showinfo("Success", message)
                elif msg_type == "DONE":
                    self.is_running = False
                    self.convert_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.progress.stop()
                    self.progress_var.set("Ready")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_output_queue)

def main():
    # Try to set a modern theme
    root = tk.Tk()
    
    try:
        # Try to use a modern theme if available
        style = ttk.Style()
        available_themes = style.theme_names()
        
        # Prefer modern themes
        preferred_themes = ['winnative', 'vista', 'xpnative', 'clam', 'alt']
        for theme in preferred_themes:
            if theme in available_themes:
                style.theme_use(theme)
                break
    except:
        pass  # Fall back to default theme
    
    app = FP8ConverterGUI(root)
    
    # Handle window close
    def on_closing():
        if app.is_running:
            if messagebox.askokcancel("Quit", "Conversion is running. Do you want to stop it and quit?"):
                app.stop_conversion()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()