class CatLitterCaptcha {
    constructor() {
        this.litterBox = document.getElementById('catLitterBox');
        this.resetBtn = document.getElementById('resetBtn');
        this.verifyBtn = document.getElementById('verifyBtn');
        this.messageDiv = document.getElementById('message');
        this.clickCountSpan = document.getElementById('clickCount');
        this.totalClumpsSpan = document.getElementById('totalClumps');
        
        this.userClicks = [];
        this.currentClumps = [];
        this.removedClumps = new Set();
        
        this.init();
    }
    
    init() {
        this.loadCaptcha();
        this.bindEvents();
    }
    
    async loadCaptcha() {
        try {
            const response = await fetch('/cat/generate');
            const data = await response.json();
            
            this.currentClumps = data.clumps || [];
            this.userClicks = [];
            this.removedClumps = new Set();
            
            this.renderCaptcha(data);
            this.updateCounter();
            this.clearMessage();
            
        } catch (error) {
            console.error('加载验证码失败:', error);
            this.showMessage('加载验证码失败，请重试', 'error');
        }
    }
    
    renderCaptcha(data) {
        // 清空猫砂盆
        this.litterBox.innerHTML = '';
        
        // 渲染块状物
        data.clumps.forEach(clump => {
            const clumpElement = document.createElement('div');
            clumpElement.className = 'clump';
            clumpElement.style.left = `${clump.x}px`;
            clumpElement.style.top = `${clump.y}px`;
            clumpElement.style.width = `${clump.size}px`;
            clumpElement.style.height = `${clump.size}px`;
            clumpElement.dataset.id = clump.id;
            
            this.litterBox.appendChild(clumpElement);
        });
        
        // 渲染干扰物
        data.distractions.forEach(distraction => {
            const distractionElement = document.createElement('div');
            distractionElement.className = `distraction ${distraction.type}`;
            distractionElement.style.left = `${distraction.x}px`;
            distractionElement.style.top = `${distraction.y}px`;
            distractionElement.title = '这是干扰物，不要点击！';
            
            this.litterBox.appendChild(distractionElement);
        });
        
        this.totalClumpsSpan.textContent = data.clumps.length;
    }
    
    bindEvents() {
        // 猫砂盆点击事件
        this.litterBox.addEventListener('click', (e) => {
            this.handleLitterBoxClick(e);
        });
        
        // 重置按钮
        this.resetBtn.addEventListener('click', () => {
            this.loadCaptcha();
        });
        
        // 验证按钮
        this.verifyBtn.addEventListener('click', () => {
            this.verifyCaptcha();
        });
    }
    
    handleLitterBoxClick(e) {
        const rect = this.litterBox.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // 添加点击效果
        this.addClickEffect(x, y);
        
        // 记录点击位置
        this.userClicks.push({ x, y });
        
        // 检查是否点击了块状物
        const clickedElement = e.target;
        if (clickedElement.classList.contains('clump')) {
            const clumpId = clickedElement.dataset.id;
            if (!this.removedClumps.has(clumpId)) {
                this.removedClumps.add(clumpId);
                clickedElement.classList.add('removed');
                this.updateCounter();
            }
        }
        
        console.log('点击位置:', { x, y });
    }
    
    addClickEffect(x, y) {
        const effect = document.createElement('div');
        effect.className = 'click-effect';
        effect.style.left = `${x - 5}px`;
        effect.style.top = `${y - 5}px`;
        this.litterBox.appendChild(effect);
        
        setTimeout(() => {
            effect.remove();
        }, 600);
    }
    
    updateCounter() {
        this.clickCountSpan.textContent = this.removedClumps.size;
    }
    
    clearMessage() {
        this.messageDiv.textContent = '';
        this.messageDiv.className = 'message';
    }
    
    showMessage(message, type = 'info') {
        this.messageDiv.textContent = message;
        this.messageDiv.className = `message ${type}`;
    }
    
    async verifyCaptcha() {
        if (this.userClicks.length === 0) {
            this.showMessage('请先点击块状物', 'error');
            return;
        }
        
        try {
            this.verifyBtn.disabled = true;
            this.verifyBtn.textContent = '验证中...';
            
            const response = await fetch('/cat/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    clicks: this.userClicks
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage('验证成功！', 'success');
                // 可以在这里添加跳转逻辑
                setTimeout(() => {
                    window.location.href = '/audio'; // 跳转到下一个验证码
                }, 1500);
            } else {
                this.showMessage('验证失败，请重试', 'error');
                setTimeout(() => {
                    this.loadCaptcha();
                }, 2000);
            }
            
        } catch (error) {
            console.error('验证失败:', error);
            this.showMessage('验证失败，请重试', 'error');
        } finally {
            this.verifyBtn.disabled = false;
            this.verifyBtn.textContent = '验证';
        }
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new CatLitterCaptcha();
});