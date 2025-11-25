class CatLitterCaptcha {
    constructor() {
        this.canvas = document.getElementById('catLitterCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.trashBin = document.getElementById('trashBin');
        this.resetBtn = document.getElementById('resetBtn');
        this.verifyBtn = document.getElementById('verifyBtn');
        this.messageDiv = document.getElementById('message');
        this.removedCountSpan = document.getElementById('removedCount');
        this.totalClumpsSpan = document.getElementById('totalClumps');
        
        this.totalClumps = 3;
        this.removedClumps = 0;
        this.clumps = [];
        this.mouseX = 0;
        this.mouseY = 0;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.isMouseDown = false;
        this.draggingClump = null;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.animationId = null;
        
        // 视觉雪纹理
        this.noiseData = null;
        this.currentNoiseData = null;
        this.baseNoiseData = null;
        
        // 性能优化
        this.frameCount = 0;
        this.updateFrequency = 2;
        
        this.init();
    }
    
    init() {
        this.generateNoiseTexture();
        this.generateClumps();
        this.bindEvents();
        this.updateCounter();
        this.animate();
    }
    
    generateNoiseTexture() {
        // 创建基础噪点纹理
        this.baseNoiseData = this.ctx.createImageData(this.canvas.width, this.canvas.height);
        this.currentNoiseData = this.ctx.createImageData(this.canvas.width, this.canvas.height);
        
        const data = this.baseNoiseData.data;
        
        // 生成密集的黑白灰噪点
        for (let i = 0; i < data.length; i += 4) {
            const value = Math.floor(Math.random() * 256);
            data[i] = value;
            data[i + 1] = value;
            data[i + 2] = value;
            data[i + 3] = 255;
        }
        
        this.currentNoiseData.data.set(this.baseNoiseData.data);
    }
    
    generateClumps() {
        this.clumps = [];
        this.removedClumps = 0;
        
        for (let i = 0; i < this.totalClumps; i++) {
            let validPosition = false;
            let x, y, width, height;
            
            while (!validPosition) {
                width = 40 + Math.random() * 35;
                height = 30 + Math.random() * 40;
                x = 30 + Math.random() * (this.canvas.width - width - 60);
                y = 30 + Math.random() * (this.canvas.height - height - 60);
                validPosition = this.isPositionValid(x, y, width, height);
            }
            
            this.clumps.push({
                id: i,
                x: x,
                y: y,
                width: width,
                height: height,
                removed: false,
                dragging: false,
                noiseData: this.generateClumpNoise(width, height)
            });
        }
        
        this.totalClumpsSpan.textContent = this.totalClumps;
    }
    
    generateClumpNoise(width, height) {
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = width;
        offscreenCanvas.height = height;
        const offscreenCtx = offscreenCanvas.getContext('2d');
        
        const imageData = offscreenCtx.createImageData(width, height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const value = Math.floor(Math.random() * 256);
            data[i] = value;
            data[i + 1] = value;
            data[i + 2] = value;
            data[i + 3] = 255;
        }
        
        offscreenCtx.putImageData(imageData, 0, 0);
        return offscreenCanvas;
    }
    
    isPositionValid(newX, newY, newWidth, newHeight) {
        for (const clump of this.clumps) {
            if (this.rectanglesOverlap(
                newX, newY, newWidth, newHeight,
                clump.x, clump.y, clump.width, clump.height
            )) {
                return false;
            }
        }
        return true;
    }
    
    rectanglesOverlap(x1, y1, w1, h1, x2, y2, w2, h2) {
        return !(x1 + w1 < x2 || x2 + w2 < x1 || y1 + h1 < y2 || y2 + h2 < y1);
    }
    
    bindEvents() {
        // 鼠标移动 - 使用document而不是canvas，这样可以在整个页面拖动
        document.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.lastMouseX = this.mouseX;
            this.lastMouseY = this.mouseY;
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
            
            // 更新拖动中的块状物位置
            if (this.draggingClump) {
                // 使用页面坐标而不是canvas相对坐标
                this.draggingClump.x = e.clientX - this.dragOffsetX - rect.left;
                this.draggingClump.y = e.clientY - this.dragOffsetY - rect.top;
                
                // 检查是否在垃圾桶上方
                this.checkTrashBinHover(e);
            }
        });
        
        // 鼠标按下
        this.canvas.addEventListener('mousedown', (e) => {
            this.isMouseDown = true;
            this.startDrag(e);
        });
        
        // 鼠标释放 - 使用document而不是canvas
        document.addEventListener('mouseup', (e) => {
            this.isMouseDown = false;
            this.endDrag(e);
        });
        
        this.resetBtn.addEventListener('click', () => {
            this.reset();
        });
        
        this.verifyBtn.addEventListener('click', () => {
            this.verifyCaptcha();
        });
    }
    
    startDrag(e) {
        const rect = this.canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        // 检查是否点击了块状物
        for (const clump of this.clumps) {
            if (clump.removed || clump.dragging) continue;
            
            if (clickX >= clump.x && clickX <= clump.x + clump.width &&
                clickY >= clump.y && clickY <= clump.y + clump.height) {
                
                // 开始拖动这个块状物
                this.draggingClump = clump;
                clump.dragging = true;
                
                // 计算拖动偏移量（使用页面坐标）
                this.dragOffsetX = e.clientX - rect.left - clump.x;
                this.dragOffsetY = e.clientY - rect.top - clump.y;
                
                // 改变鼠标样式
                document.body.style.cursor = 'grabbing';
                break;
            }
        }
    }
    
    checkTrashBinHover(e) {
        if (!this.draggingClump) return;
        
        const trashRect = this.trashBin.getBoundingClientRect();
        const isOverTrash = 
            e.clientX >= trashRect.left && 
            e.clientX <= trashRect.right &&
            e.clientY >= trashRect.top && 
            e.clientY <= trashRect.bottom;
        
        if (isOverTrash) {
            this.trashBin.classList.add('active');
        } else {
            this.trashBin.classList.remove('active');
        }
    }
    
    endDrag(e) {
        if (!this.draggingClump) {
            document.body.style.cursor = 'default';
            return;
        }
        
        // 恢复鼠标样式
        document.body.style.cursor = 'default';
        
        // 检查是否在垃圾桶上方
        const trashRect = this.trashBin.getBoundingClientRect();
        const isOverTrash = 
            e.clientX >= trashRect.left && 
            e.clientX <= trashRect.right &&
            e.clientY >= trashRect.top && 
            e.clientY <= trashRect.bottom;
        
        if (isOverTrash) {
            // 成功移除块状物
            this.removeClump(this.draggingClump);
        } else {
            // 放回猫砂盆中
            this.returnClump(this.draggingClump);
        }
        
        this.draggingClump.dragging = false;
        this.draggingClump = null;
        this.trashBin.classList.remove('active');
    }
    
    returnClump(clump) {
        let validPosition = false;
        let newX, newY;
        
        for (let attempt = 0; attempt < 20 && !validPosition; attempt++) {
            newX = 30 + Math.random() * (this.canvas.width - clump.width - 60);
            newY = 30 + Math.random() * (this.canvas.height - clump.height - 60);
            validPosition = this.isPositionValid(newX, newY, clump.width, clump.height);
        }
        
        if (validPosition) {
            clump.x = newX;
            clump.y = newY;
        } else {
            clump.x = 30 + Math.random() * (this.canvas.width - clump.width - 60);
            clump.y = 30 + Math.random() * (this.canvas.height - clump.height - 60);
        }
    }
    
    removeClump(clump) {
        clump.removed = true;
        this.removedClumps++;
        
        this.showRemoveEffect();
        this.updateCounter();
    }
    
    showRemoveEffect() {
        this.trashBin.classList.add('receiving');
        setTimeout(() => {
            this.trashBin.classList.remove('receiving');
        }, 500);
    }
    
    animate() {
        this.frameCount++;
        
        // 绘制基础噪点纹理
        this.ctx.putImageData(this.currentNoiseData, 0, 0);
        
        // 定期更新噪点纹理，产生流动效果
        if (this.frameCount % this.updateFrequency === 0) {
            this.updateNoiseTexture();
        }
        
        // 绘制块状物
        this.drawClumps();
        
        // 绘制被拖动的块状物（在最上层）
        if (this.draggingClump) {
            this.drawDraggingClump();
        }
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    updateNoiseTexture() {
        this.currentNoiseData.data.set(this.baseNoiseData.data);
        
        const data = this.currentNoiseData.data;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        const mouseSpeed = Math.sqrt(
            Math.pow(this.mouseX - this.lastMouseX, 2) + 
            Math.pow(this.mouseY - this.lastMouseY, 2)
        );
        
        if (mouseSpeed > 0.5) {
            const influenceRadius = 60 + mouseSpeed * 5;
            const distortionStrength = Math.min(10, mouseSpeed * 2);
            
            for (let y = Math.max(0, this.mouseY - influenceRadius); 
                 y < Math.min(height, this.mouseY + influenceRadius); y++) {
                
                for (let x = Math.max(0, this.mouseX - influenceRadius); 
                     x < Math.min(width, this.mouseX + influenceRadius); x++) {
                    
                    const distance = Math.sqrt(
                        Math.pow(x - this.mouseX, 2) + 
                        Math.pow(y - this.mouseY, 2)
                    );
                    
                    if (distance < influenceRadius) {
                        let inClump = false;
                        for (const clump of this.clumps) {
                            if (clump.removed || clump.dragging) continue;
                            
                            if (x >= clump.x && x <= clump.x + clump.width &&
                                y >= clump.y && y <= clump.y + clump.height) {
                                inClump = true;
                                break;
                            }
                        }
                        
                        if (!inClump) {
                            const force = (influenceRadius - distance) / influenceRadius;
                            const angle = Math.atan2(y - this.mouseY, x - this.mouseX);
                            
                            const distortion = force * distortionStrength;
                            const sourceX = Math.floor(x + Math.cos(angle) * distortion);
                            const sourceY = Math.floor(y + Math.sin(angle) * distortion);
                            
                            if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height) {
                                const currentIndex = (y * width + x) * 4;
                                const sourceIndex = (sourceY * width + sourceX) * 4;
                                
                                for (let i = 0; i < 4; i++) {
                                    data[currentIndex + i] = this.baseNoiseData.data[sourceIndex + i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    drawClumps() {
        this.clumps.forEach(clump => {
            if (clump.removed || clump.dragging) return;
            
            this.ctx.drawImage(
                clump.noiseData, 
                clump.x, 
                clump.y, 
                clump.width, 
                clump.height
            );
            
            const distance = Math.sqrt(
                Math.pow(this.mouseX - (clump.x + clump.width/2), 2) +
                Math.pow(this.mouseY - (clump.y + clump.height/2), 2)
            );
            
            if (distance < 80) {
                this.ctx.strokeStyle = `rgba(255, 255, 255, ${0.3 * (1 - distance/80)})`;
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(
                    clump.x - 1, 
                    clump.y - 1, 
                    clump.width + 2, 
                    clump.height + 2
                );
            }
        });
    }
    
    drawDraggingClump() {
        if (!this.draggingClump) return;
        
        this.ctx.drawImage(
            this.draggingClump.noiseData, 
            this.draggingClump.x, 
            this.draggingClump.y, 
            this.draggingClump.width, 
            this.draggingClump.height
        );
        
        this.ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(
            this.draggingClump.x - 2, 
            this.draggingClump.y - 2, 
            this.draggingClump.width + 4, 
            this.draggingClump.height + 4
        );
        this.ctx.setLineDash([]);
    }
    
    updateCounter() {
        this.removedCountSpan.textContent = this.removedClumps;
    }
    
    clearMessage() {
        this.messageDiv.textContent = '';
        this.messageDiv.className = 'message';
    }
    
    reset() {
        this.generateNoiseTexture();
        this.generateClumps();
        this.updateCounter();
        this.clearMessage();
        this.draggingClump = null;
        document.body.style.cursor = 'default';
    }
    
    async verifyCaptcha() {
        if (this.removedClumps < this.totalClumps) {
            this.messageDiv.textContent = `还需要移除 ${this.totalClumps - this.removedClumps} 个块状物`;
            this.messageDiv.className = 'message error';
            return;
        }
        
        try {
            const response = await fetch('/cat/log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: 'cat_litter_cleaned',
                    removed: this.removedClumps,
                    total: this.totalClumps,
                    timestamp: new Date().toISOString()
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'ok') {
                this.messageDiv.textContent = '猫砂清理完成！验证成功！';
                this.messageDiv.className = 'message success';
                
                setTimeout(() => {
                    window.location.href = '/audio';
                }, 1500);
            }
            
        } catch (error) {
            console.error('验证失败:', error);
            this.messageDiv.textContent = '验证失败，请重试';
            this.messageDiv.className = 'message error';
        }
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        document.body.style.cursor = 'default';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CatLitterCaptcha();
});