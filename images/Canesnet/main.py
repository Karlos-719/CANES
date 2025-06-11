import sys
sys.path.append('../')
from pycore.tikzeng import *

# Ultra-Clean UTransMambaNet - Maximum readability with organized rows
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # ========= ROW 1: TRANSFORMER PROCESSING (TOP) =========
    
    to_Conv("trans3", 128, 256, offset="(15,8,0)", to="(0,8,0)", height=12, depth=12, width=3, caption="Transformer\\\\Block 3"),
    to_Conv("trans4", 64, 512, offset="(8,0,0)", to="(trans3-east)", height=6, depth=6, width=4, caption="Transformer\\\\Block 4"),
    
    # ========= ROW 2: FUSION MODULES (UPPER MIDDLE) =========
    
    to_Conv("fusion1", 512, 64, offset="(3,4,0)", to="(0,4,0)", height=32, depth=32, width=1.5, caption="Fusion\\\\Module 1"),
    to_Conv("fusion2", 256, 128, offset="(8,0,0)", to="(fusion1-east)", height=16, depth=16, width=1.8, caption="Fusion\\\\Module 2"),
    to_Conv("fusion3", 128, 256, offset="(8,0,0)", to="(fusion2-east)", height=8, depth=8, width=2.2, caption="Fusion\\\\Module 3"),
    to_Conv("fusion4", 64, 512, offset="(8,0,0)", to="(fusion3-east)", height=4, depth=4, width=2.8, caption="Fusion\\\\Module 4"),
    
    # ========= ROW 3: MAIN U-NET BACKBONE (CENTER) =========
    
    # Encoder
    to_Conv("input", 512, 3, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=1, caption="Input\\\\3ch"),
    to_Conv("enc1", 512, 64, offset="(4,0,0)", to="(input-east)", height=64, depth=64, width=2, caption="Encoder\\\\64"),
    to_Pool("pool1", offset="(3,0,0)", to="(enc1-east)", height=32, depth=32, width=1),
    
    to_Conv("enc2", 256, 128, offset="(3,0,0)", to="(pool1-east)", height=32, depth=32, width=2.5, caption="Encoder\\\\128"),
    to_Pool("pool2", offset="(3,0,0)", to="(enc2-east)", height=16, depth=16, width=1),
    
    to_Conv("enc3", 128, 256, offset="(3,0,0)", to="(pool2-east)", height=16, depth=16, width=3, caption="Encoder\\\\256"),
    to_Pool("pool3", offset="(3,0,0)", to="(enc3-east)", height=8, depth=8, width=1),
    
    to_Conv("enc4", 64, 512, offset="(3,0,0)", to="(pool3-east)", height=8, depth=8, width=4, caption="Encoder\\\\512"),
    to_Pool("pool4", offset="(3,0,0)", to="(enc4-east)", height=4, depth=4, width=1),
    
    # Bottleneck
    to_Conv("bottleneck", 32, 1024, offset="(3,0,0)", to="(pool4-east)", height=4, depth=4, width=5, caption="Bottleneck\\\\1024"),
    to_Conv("aspp", 32, 1024, offset="(0,0,3)", to="(bottleneck-near)", height=4, depth=4, width=5, caption="ASPP\\\\Multi-Scale"),
    
    # Decoder
    to_Conv("dec4", 64, 512, offset="(4,0,0)", to="(aspp-east)", height=8, depth=8, width=4, caption="Decoder\\\\512"),
    to_Conv("dec3", 128, 256, offset="(4,0,0)", to="(dec4-east)", height=16, depth=16, width=3, caption="Decoder\\\\256"),
    to_Conv("dec2", 256, 128, offset="(4,0,0)", to="(dec3-east)", height=32, depth=32, width=2.5, caption="Decoder\\\\128"),
    to_Conv("dec1", 512, 64, offset="(4,0,0)", to="(dec2-east)", height=64, depth=64, width=2, caption="Decoder\\\\64"),
    to_Conv("output", 512, 3, offset="(4,0,0)", to="(dec1-east)", height=64, depth=64, width=1, caption="Output\\\\3ch"),
    
    # ========= ROW 4: MAMBA PROCESSING (BOTTOM) =========
    
    to_Conv("mamba1", 512, 64, offset="(7,-8,0)", to="(0,-8,0)", height=48, depth=48, width=2, caption="Mamba\\\\Block 1"),
    to_Conv("mamba2", 256, 128, offset="(8,0,0)", to="(mamba1-east)", height=24, depth=24, width=2.5, caption="Mamba\\\\Block 2"),
    
    # ========= MAIN BACKBONE CONNECTIONS =========
    
    to_connection("input", "enc1"),
    to_connection("enc1", "pool1"),
    to_connection("pool1", "enc2"),
    to_connection("enc2", "pool2"),
    to_connection("pool2", "enc3"),
    to_connection("enc3", "pool3"),
    to_connection("pool3", "enc4"),
    to_connection("enc4", "pool4"),
    to_connection("pool4", "bottleneck"),
    to_connection("bottleneck", "aspp"),
    
    to_connection("aspp", "dec4"),
    to_connection("dec4", "dec3"),
    to_connection("dec3", "dec2"),
    to_connection("dec2", "dec1"),
    to_connection("dec1", "output"),
    
    # ========= VERTICAL BRANCH CONNECTIONS =========
    
    # From encoders to processing blocks
    to_connection("enc1", "mamba1"),
    to_connection("enc2", "mamba2"),
    to_connection("enc3", "trans3"),
    to_connection("enc4", "trans4"),
    
    # From processing blocks to fusion
    to_connection("mamba1", "fusion1"),
    to_connection("mamba2", "fusion2"),
    to_connection("trans3", "fusion3"),
    to_connection("trans4", "fusion4"),
    
    # From fusion to decoders
    to_connection("fusion1", "dec1"),
    to_connection("fusion2", "dec2"),
    to_connection("fusion3", "dec3"),
    to_connection("fusion4", "dec4"),
    
    # ========= U-NET SKIP CONNECTIONS =========
    
    to_skip("fusion4", "dec4"),
    to_skip("fusion3", "dec3"),
    to_skip("fusion2", "dec2"),
    to_skip("fusion1", "dec1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()