import torch
import torch.nn as nn
import time
import numpy as np
import os
import cv2
from net3 import Restormer_Encoder, Restormer_Decoder
from nets.unetCursor import Unet
import warnings
import json

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PerformanceAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        self.results = []
        
    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size_mb = total_params * 4 / (1024 ** 2)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'size_mb': param_size_mb
        }
    
    def count_flops(self, model, input_ir, input_vis):
        try:
            from thop import profile, clever_format
            from net3 import Restormer_Encoder, Restormer_Decoder
            from nets.unetCursor import Unet
            
            temp_fusion_encoder = Restormer_Encoder().cpu()
            temp_fusion_decoder = Restormer_Decoder().cpu()
            temp_seg_model = Unet(num_classes=4, backbone='resnet50').cpu()
            
            class E2EModel(nn.Module):
                def __init__(self, enc, dec, seg):
                    super().__init__()
                    self.encoder = enc
                    self.decoder = dec
                    self.seg = seg
                
                def forward(self, ir, vis):
                    feat_V_B, feat_V_D, _ = self.encoder(vis)
                    feat_I_B, feat_I_D, _ = self.encoder(ir)
                    fused, _ = self.decoder(vis, feat_V_B + feat_I_B, feat_V_D + feat_I_D)
                    fused = (fused - fused.min()) / (fused.max() - fused.min())
                    fused_3ch = fused.repeat(1, 3, 1, 1)
                    seg_out = self.seg(fused_3ch)
                    return fused, seg_out
            
            temp_model = E2EModel(temp_fusion_encoder, temp_fusion_decoder, temp_seg_model).cpu()
            
            input_ir_cpu = input_ir.cpu()
            input_vis_cpu = input_vis.cpu()
            
            flops, params = profile(temp_model, inputs=(input_ir_cpu, input_vis_cpu), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            
            del temp_model, temp_fusion_encoder, temp_fusion_decoder, temp_seg_model
            torch.cuda.empty_cache()
            
            return {'flops': flops, 'params': params}
            
        except ImportError:
            return {'flops': 'N/A', 'params': 'N/A'}
        except Exception as e:
            return {'flops': 'N/A', 'params': 'N/A'}
    
    def measure_memory(self, model, input_ir, input_vis, warmup=5, iterations=10):
        if self.device != 'cuda':
            return {'peak_memory_mb': 'N/A'}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ir, input_vis)
                torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_ir, input_vis)
                torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {'peak_memory_mb': peak_memory}
    
    def measure_latency(self, encoder, decoder, seg_model, input_ir, input_vis, iterations=100):
        fusion_times = []
        seg_times = []
        
        with torch.no_grad():
            for _ in range(iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                fusion_start = time.time()
                
                feat_V_B, feat_V_D, _ = encoder(input_vis)
                feat_I_B, feat_I_D, _ = encoder(input_ir)
                fused, _ = decoder(input_vis, feat_V_B + feat_I_B, feat_V_D + feat_I_D)
                fused = (fused - fused.min()) / (fused.max() - fused.min())
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                fusion_end = time.time()
                fusion_times.append((fusion_end - fusion_start) * 1000)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                seg_start = time.time()
                
                fused_3ch = fused.repeat(1, 3, 1, 1)
                _ = seg_model(fused_3ch)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                seg_end = time.time()
                seg_times.append((seg_end - seg_start) * 1000)
        
        fusion_mean = np.mean(fusion_times)
        seg_mean = np.mean(seg_times)
        total_mean = fusion_mean + seg_mean
        
        return {
            'fusion_ms': fusion_mean,
            'segmentation_ms': seg_mean,
            'total_ms': total_mean,
            'fps': 1000 / total_mean
        }
    
    def test_configuration(self, name, encoder, decoder, seg_model, input_shape, 
                          iterations=100, use_fp16=False):
        print(f"\n{'='*80}")
        print(f"Config: {name} | Resolution: {input_shape[0]}x{input_shape[1]}" + 
              (f" | Precision: FP16" if use_fp16 else ""))
        print(f"{'='*80}")
        
        class E2EModel(nn.Module):
            def __init__(self, enc, dec, seg):
                super().__init__()
                self.encoder = enc
                self.decoder = dec
                self.seg = seg
            
            def forward(self, ir, vis):
                feat_V_B, feat_V_D, _ = self.encoder(vis)
                feat_I_B, feat_I_D, _ = self.encoder(ir)
                fused, _ = self.decoder(vis, feat_V_B + feat_I_B, feat_V_D + feat_I_D)
                fused = (fused - fused.min()) / (fused.max() - fused.min())
                fused_3ch = fused.repeat(1, 3, 1, 1)
                seg_out = self.seg(fused_3ch)
                return fused, seg_out
        
        model = E2EModel(encoder, decoder, seg_model).to(self.device).eval()
        
        test_ir = torch.randn(1, 1, input_shape[0], input_shape[1]).to(self.device)
        test_vis = torch.randn(1, 1, input_shape[0], input_shape[1]).to(self.device)
        
        if use_fp16:
            test_ir = test_ir.half()
            test_vis = test_vis.half()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_ir, test_vis)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        
        params = self.count_parameters(model)
        print(f"Params: {params['total']/1e6:.2f}M | Size: {params['size_mb']:.2f}MB")
        
        memory_result = self.measure_memory(model, test_ir, test_vis)
        if memory_result['peak_memory_mb'] != 'N/A':
            print(f"Peak Memory: {memory_result['peak_memory_mb']:.2f}MB")
        
        latency_result = self.measure_latency(encoder, decoder, seg_model, test_ir, test_vis, iterations)
        
        flops_result = {'flops': 'N/A', 'params': 'N/A'}
        if not use_fp16:
            flops_result = self.count_flops(model, test_ir, test_vis)
            print(f"FLOPs: {flops_result['flops']}")
        
        result = {
            'name': name,
            'resolution': f"{input_shape[0]}x{input_shape[1]}",
            'params_m': params['total'] / 1e6,
            'size_mb': params['size_mb'],
            'flops': flops_result['flops'] if not use_fp16 else 'N/A',
            'memory_mb': memory_result['peak_memory_mb'],
            'fusion_ms': latency_result['fusion_ms'],
            'segmentation_ms': latency_result['segmentation_ms'],
            'total_ms': latency_result['total_ms'],
            'fps': latency_result['fps']
        }
        
        self.results.append(result)
        
        print(f"\nPerformance:")
        print(f"  Fusion: {latency_result['fusion_ms']:6.2f}ms ({latency_result['fusion_ms']/latency_result['total_ms']*100:.1f}%)")
        print(f"  Segmentation: {latency_result['segmentation_ms']:6.2f}ms ({latency_result['segmentation_ms']/latency_result['total_ms']*100:.1f}%)")
        print(f"  Total: {latency_result['total_ms']:6.2f}ms | FPS: {latency_result['fps']:6.2f}")
        
        del model
        torch.cuda.empty_cache()
        
        return result
    
    def print_summary(self):
        print(f"\n\n{'='*120}")
        print("Performance Comparison")
        print(f"{'='*120}")
        
        print(f"\n{'Config':<25} {'Resolution':<10} {'Params(M)':<10} {'FLOPs':<12} {'Memory(MB)':<10} "
              f"{'Fusion(ms)':<10} {'Seg(ms)':<10} {'Total(ms)':<10} {'FPS':<8} {'Speedup':<8}")
        print("-" * 120)
        
        baseline = self.results[0] if self.results else None
        
        for r in self.results:
            speedup = ""
            if baseline and r != baseline:
                speedup = f"{baseline['total_ms']/r['total_ms']:.2f}x"
            
            flops_str = str(r['flops']) if r['flops'] != 'N/A' else 'N/A'
            memory_str = f"{r['memory_mb']:.1f}" if r['memory_mb'] != 'N/A' else 'N/A'
            
            print(f"{r['name']:<25} {r['resolution']:<10} {r['params_m']:<10.2f} {flops_str:<12} "
                  f"{memory_str:<10} {r['fusion_ms']:<10.2f} {r['segmentation_ms']:<10.2f} "
                  f"{r['total_ms']:<10.2f} {r['fps']:<8.2f} {speedup:<8}")
        
        print(f"{'='*120}\n")


def main(use_fp16_test=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"{'='*80}")
    
    analyzer = PerformanceAnalyzer(device)
    
    encoder_512 = nn.DataParallel(Restormer_Encoder()).to(device).eval()
    decoder_512 = nn.DataParallel(Restormer_Decoder()).to(device).eval()
    seg_512 = Unet(num_classes=4, backbone='resnet50').to(device).eval()
    
    analyzer.test_configuration(
        "Baseline (FP32)",
        encoder_512, decoder_512, seg_512,
        [512, 512],
        iterations=50,
        use_fp16=False
    )
    
    encoder_256 = nn.DataParallel(Restormer_Encoder()).to(device).eval()
    decoder_256 = nn.DataParallel(Restormer_Decoder()).to(device).eval()
    seg_256 = Unet(num_classes=4, backbone='resnet50').to(device).eval()
    
    analyzer.test_configuration(
        "Low-Res (FP32)",
        encoder_256, decoder_256, seg_256,
        [256, 256],
        iterations=50,
        use_fp16=False
    )
    
    if use_fp16_test:
        try:
            encoder_fp16_512 = nn.DataParallel(Restormer_Encoder()).to(device).half().eval()
            decoder_fp16_512 = nn.DataParallel(Restormer_Decoder()).to(device).half().eval()
            seg_fp16_512 = Unet(num_classes=4, backbone='resnet50').to(device).half().eval()
            
            analyzer.test_configuration(
                "Mixed-Precision (FP16)",
                encoder_fp16_512, decoder_fp16_512, seg_fp16_512,
                [512, 512],
                iterations=50,
                use_fp16=True
            )
        except Exception as e:
            print(f"\n⚠ FP16 test failed: {e}")
        
        try:
            encoder_fp16_256 = nn.DataParallel(Restormer_Encoder()).to(device).half().eval()
            decoder_fp16_256 = nn.DataParallel(Restormer_Decoder()).to(device).half().eval()
            seg_fp16_256 = Unet(num_classes=4, backbone='resnet50').to(device).half().eval()
            
            analyzer.test_configuration(
                "FP16 + Low-Res",
                encoder_fp16_256, decoder_fp16_256, seg_fp16_256,
                [256, 256],
                iterations=50,
                use_fp16=True
            )
        except Exception as e:
            print(f"\n⚠ FP16 + Low-Res test failed: {e}")
    
    analyzer.print_summary()
    
    os.makedirs('test_result', exist_ok=True)
    results_file = 'test_result/fps_analysis.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'hardware': {
                'device': device,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A',
                'pytorch_version': torch.__version__
            },
            'results': analyzer.results
        }, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FPS Performance Test')
    parser.add_argument('--no-fp16', action='store_true', help='Skip FP16 tests')
    
    args = parser.parse_args()
    
    try:
        main(use_fp16_test=not args.no_fp16)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

