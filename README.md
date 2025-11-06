# 🧩 Diffusion Denoising Process Visualization

These images show the **reverse diffusion trajectory** — how the model progressively *denoises* a noisy sample back into a clear image.  
They likely come from the **DDpM** implementation using the **`UNetBase`** architecture.

---

## 🌀 Step 717 — Early Stage
![x0_717](https://github.com/user-attachments/assets/a24a53e7-abe6-443d-b59f-eaae5287d272)

- At this point, the image is **almost pure noise**.  
- The diffusion process starts from a Gaussian noise sample \( x_T \).  
- There’s no visible structure yet — the model only begins to learn statistical patterns of what the final image should look like.  

**Role of U-Net:**  
- The **encoder** path captures global patterns and context, even in noisy inputs.  
- The **time embedding** informs the model that it’s at an early step (high noise level), guiding it to make only coarse denoising corrections.

---

## 🪄 Step 225 — Mid Stage
![x0_225](https://github.com/user-attachments/assets/88dde179-2cc2-4279-a768-8941eaf9b967)

- Now, **basic shapes and colors** start emerging.  
- The model has gradually reduced high-frequency noise and started constructing meaningful structure.  
- Details are still blurry or inconsistent — the network is refining global structure but hasn’t yet restored texture.

**Role of U-Net:**  
- **Skip connections** help retain spatial coherence and recover details lost during downsampling.  
- The **decoder** begins reconstructing fine details based on features preserved by the encoder.

---

## ✨ Step 0 — Final Denoised Output
![x0_0](https://github.com/user-attachments/assets/ed2e738b-6907-4cf4-bfcd-cabc01549321)

- This is the **final reconstructed image**, after all diffusion steps are reversed.  
- Most noise has been removed, and the model’s prediction \( \hat{x}_0 \) closely approximates the true image.  
- The U-Net has effectively combined global context and fine details to restore edges, texture, and overall structure.

**Role of U-Net:**  
- The **decoder** outputs the clean image prediction.  
- **Skip connections** ensure high-frequency details are preserved.  
- **Time conditioning** ensures the model knows how much denoising to perform.

---

### 🧠 Summary

| Stage | Noise Level | Visual Appearance | U-Net Role |
|:------|:-------------|:------------------|:------------|
| Step 717 | Very High | Pure noise, no structure | Encoder captures context, time embedding guides coarse denoising |
| Step 225 | Medium | Shapes emerging, blurry textures | Skip connections restore structure |
| Step 0 | Low | Sharp, coherent final image | Decoder reconstructs fine details |

---

📘 **Takeaway:**  
The U-Net architecture is ideal for diffusion models because it can both **capture global context** (via downsampling) and **preserve fine details** (via skip connections), while **time conditioning** controls the denoising strength at each step.
