import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
import subprocess
from pathlib import Path

class FP8QuantizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FP8 Quantization Tool")
        self.root.geometry("800x700")
        self.root.minsize(600, 500)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.t5xxl = tk.BooleanVar()
        self.keep_distillation = tk.BooleanVar()
        self.calib_samples = tk.IntVar(value=3072)
        self.num_iter = tk.IntVar(value=500)
        self.top_k = tk.IntVar(value=1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="FP8 Quantization Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Input file
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        input_entry = ttk.Entry(file_frame, textvariable=self.input_file, width=60)
        input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=(0, 5))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_input_file).grid(row=0, column=2, pady=(0, 5))
        
        # Output file
        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        output_entry = ttk.Entry(file_frame, textvariable=self.output_file, width=60)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=(0, 5))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_output_file).grid(row=1, column=2, pady=(0, 5))
        
        # Auto-generate output filename button
        ttk.Button(file_frame, text="Auto-Generate Output Name", 
                  command=self.auto_generate_output).grid(row=2, column=1, pady=(10, 0))
        
        # Options Section
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Checkboxes
        ttk.Checkbutton(options_frame, text="T5XXL Mode", 
                       variable=self.t5xxl).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Keep Distillation Layers", 
                       variable=self.keep_distillation).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Parameters Section
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)
        
        # Calibration samples
        ttk.Label(params_frame, text="Calibration Samples:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        calib_spin = ttk.Spinbox(params_frame, from_=512, to=10000, increment=512, 
                                textvariable=self.calib_samples, width=10)
        calib_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Number of iterations
        ttk.Label(params_frame, text="Iterations:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        iter_spin = ttk.Spinbox(params_frame, from_=50, to=2000, increment=50, 
                               textvariable=self.num_iter, width=10)
        iter_spin.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Top K
        ttk.Label(params_frame, text="Top K:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        topk_spin = ttk.Spinbox(params_frame, from_=1, to=10, increment=1, 
                               textvariable=self.top_k, width=10)
        topk_spin.grid(row=0, column=5, sticky=tk.W)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to start...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(progress_frame, height=15, width=80)
        self.output_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        
        self.convert_button = ttk.Button(button_frame, text="Start Conversion", 
                                        command=self.start_conversion, style='Accent.TButton')
        self.convert_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_conversion, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Log", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.LEFT)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.process = None
        
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input Safetensors File",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            if not self.output_file.get():
                self.auto_generate_output()
    
    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output File Location",
            defaultextension=".safetensors",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
    
    def auto_generate_output(self):
        input_path = self.input_file.get()
        if not input_path:
            return
            
        base_name = os.path.splitext(input_path)[0]
        distill_str = "_nodistill" if self.keep_distillation.get() else ""
        output_path = f"{base_name}_float8_e4m3fn_scaled_learned{distill_str}_svd.safetensors"
        self.output_file.set(output_path)
    
    def clear_log(self):
        self.output_text.delete(1.0, tk.END)
    
    def log_message(self, message):
        """Add message to the output text area"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def validate_inputs(self):
        """Validate user inputs before starting conversion"""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file.")
            return False
            
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return False
            
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file.")
            return False
            
        if os.path.abspath(self.input_file.get()) == os.path.abspath(self.output_file.get()):
            messagebox.showerror("Error", "Output file cannot be the same as input file.")
            return False
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_file.get())
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {e}")
                return False
                
        return True
    
    def start_conversion(self):
        """Start the conversion process in a separate thread"""
        if not self.validate_inputs():
            return
            
        # Disable the convert button and enable stop button
        self.convert_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Clear previous output
        self.clear_log()
        self.progress_var.set("Starting conversion...")
        
        # Start conversion in a separate thread
        self.conversion_thread = threading.Thread(target=self.run_conversion)
        self.conversion_thread.daemon = True
        self.conversion_thread.start()
    
    def run_conversion(self):
        """Run the actual conversion process"""
        try:
            # Build command line arguments
            script_path = "convert_fp8_scaled_learned_svd_fast2.py"
            cmd = [sys.executable, script_path]
            
            cmd.extend(["--input", self.input_file.get()])
            cmd.extend(["--output", self.output_file.get()])
            cmd.extend(["--calib_samples", str(self.calib_samples.get())])
            cmd.extend(["--num_iter", str(self.num_iter.get())])
            cmd.extend(["--top_k", str(self.top_k.get())])
            
            if self.t5xxl.get():
                cmd.append("--t5xxl")
            if self.keep_distillation.get():
                cmd.append("--keep_distillation")
            
            self.log_message(f"Running command: {' '.join(cmd)}")
            self.log_message("-" * 80)
            
            # Run the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line
            while True:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    # Update GUI in main thread
                    self.root.after(0, self.log_message, output.strip())
            
            # Wait for process to complete
            return_code = self.process.poll()
            
            if return_code == 0:
                self.root.after(0, self.progress_var.set, "Conversion completed successfully!")
                self.root.after(0, self.log_message, "\n✓ Conversion completed successfully!")
                self.root.after(0, messagebox.showinfo, "Success", "Conversion completed successfully!")
            else:
                self.root.after(0, self.progress_var.set, "Conversion failed!")
                self.root.after(0, self.log_message, f"\n✗ Conversion failed with return code {return_code}")
                self.root.after(0, messagebox.showerror, "Error", "Conversion failed. Check the log for details.")
                
        except FileNotFoundError:
            error_msg = f"Could not find the conversion script: {script_path}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Error", error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, messagebox.showerror, "Error", error_msg)
        finally:
            # Re-enable buttons
            self.root.after(0, self.convert_button.config, {'state': 'normal'})
            self.root.after(0, self.stop_button.config, {'state': 'disabled'})
            self.process = None
    
    def stop_conversion(self):
        """Stop the conversion process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.log_message("\n⚠ Conversion stopped by user.")
            self.progress_var.set("Conversion stopped.")
            
        self.convert_button.config(state='normal')
        self.stop_button.config(state='disabled')

def main():
    root = tk.Tk()
    
    # Configure style for better appearance
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    
    app = FP8QuantizationGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()