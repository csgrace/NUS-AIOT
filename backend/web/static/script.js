const API_BASE_URL = 'http://192.168.137.14:5000';

document.addEventListener('DOMContentLoaded', () => {
    // 侧边栏导航
    const navItems = document.querySelectorAll('.nav-item');
    const navLinks = document.querySelectorAll('.submenu li[data-content]');
    const sections = document.querySelectorAll('.content-section, .welcome-message');
    const settingsNavHeader = document.querySelector('#settings-nav .nav-header');
    const contentTitle = document.getElementById('content-title'); // 获取标题元素

    if (settingsNavHeader) {
        settingsNavHeader.addEventListener('click', function () {
            document.querySelectorAll('.content-section, .welcome-message').forEach(sec => sec.classList.remove('active'));
            document.getElementById('settings-panel')?.classList.add('active');
        });
    }
    navItems.forEach(item => {
        const header = item.querySelector('.nav-header');
        if (!header) return; // 防止header为null
        const arrow = header.querySelector('.arrow');
        header.addEventListener('click', function(e) {
            e.stopPropagation();
            navItems.forEach(otherItem => {
                if (otherItem !== item && otherItem.classList.contains('active')) {
                    otherItem.classList.remove('active');
                    const otherArrow = otherItem.querySelector('.arrow');
                    if (otherArrow && otherArrow.classList.contains('fa-chevron-down')) {
                        otherArrow.classList.remove('fa-chevron-down');
                        otherArrow.classList.add('fa-chevron-right');
                    }
                }
            });
            item.classList.toggle('active');
            if (arrow) { // 只处理有arrow的header
                if (item.classList.contains('active')) {
                    arrow.classList.remove('fa-chevron-right');
                    arrow.classList.add('fa-chevron-down');
                } else {
                    arrow.classList.remove('fa-chevron-down');
                    arrow.classList.add('fa-chevron-right');
                }
            }
        });
    });
    navLinks.forEach(link => {
        link.addEventListener('click', function () {
            sections.forEach(sec => sec.classList.remove('active'));
            const id = link.getAttribute('data-content');
            const showSection = document.getElementById(id);
            if (showSection) showSection.classList.add('active');
            // 新增：如果是workout-mode，调用检测函数
            if (id === 'workout-mode') {
                handleEnterWorkoutMode();
            }
        });
    });
    document.querySelectorAll('.nav-header[data-content]').forEach(header => {
        header.addEventListener('click', function () {
            sections.forEach(sec => sec.classList.remove('active'));
            const id = header.getAttribute('data-content');
            const showSection = document.getElementById(id);
            if (showSection) showSection.classList.add('active');
            if (id === 'workout-mode') {
                handleEnterWorkoutMode();
            }
        });
    });    

    /*/===== =  = = === = ===== ===== =   HOME    === ====== ============== ======= ===/*/

    // // 步数
    async function refreshStepsAndHome() {
        try {
            const res = await fetch(`${API_BASE_URL}/getSteps`);
            const data = await res.json();
            const steps = Number(data.steps) || 0;
            // 原界面
            const stepCountEl = document.getElementById('step-count');
            if (stepCountEl) stepCountEl.textContent = data.steps ?? '--';
            // Home Section
            const homeStepEl = document.getElementById('home-step-count');
            if (homeStepEl) homeStepEl.textContent = data.steps ?? '--';
            window.latestSteps = steps;
            // calculateCaloriesBurned && calculateCaloriesBurned();
        } catch {
            const stepCountEl = document.getElementById('step-count');
            if (stepCountEl) stepCountEl.textContent = 'ga';
            const homeStepEl = document.getElementById('home-step-count');
            if (homeStepEl) homeStepEl.textContent = 'ga';
            console.error('Failed to fetch steps:', err);
        }
    }
    // // 心率
    async function refreshHeartRateAndHome() {
        try {
            const res = await fetch(`${API_BASE_URL}/getHeartRate`);
            const data = await res.json();
            // 原界面
            const heartRateEl = document.getElementById('heart-rate-value');
            if (heartRateEl) heartRateEl.textContent = data.value ?? '--';
            // Home Section
            const homeHrEl = document.getElementById('home-hr-value');
            if (homeHrEl) homeHrEl.textContent = data.value ?? '--';
        } catch {
            const heartRateEl = document.getElementById('heart-rate-value');
            if (heartRateEl) heartRateEl.textContent = '--';
            const homeHrEl = document.getElementById('home-hr-value');
            if (homeHrEl) homeHrEl.textContent = '--';
        }
    }
    //
    // // 喝水
    async function refreshHomeWaterToday() {
        const dateStr = (new Date()).toISOString().substring(0, 10);
        try {
            const res = await fetch(`${API_BASE_URL}/getWaterAmount?date=${encodeURIComponent(dateStr)}`);
            const data = await res.json();
            const homeWaterTodayEl = document.getElementById('home-water-today');
            if (homeWaterTodayEl) homeWaterTodayEl.textContent = data.amount ? (data.amount + 'ml') : '--';
        } catch {
            const homeWaterTodayEl = document.getElementById('home-water-today');
            if (homeWaterTodayEl) homeWaterTodayEl.textContent = '--';
        }
    }
    // 摄入卡路里（consumed）
    async function refreshCaloriesConsumedHome() {
        const dateStr = (new Date()).toISOString().substring(0, 10);
        try {
            const res = await fetch(`${API_BASE_URL}/getCaloriesConsumed?date=${encodeURIComponent(dateStr)}`);
            const data = await res.json();
            const homeCalorieConsumedEl = document.getElementById('home-calorie-consumed');
            if (homeCalorieConsumedEl)
                homeCalorieConsumedEl.textContent = (typeof data.calories === 'number') ? data.calories.toFixed(2) + ' kcal' : '--';
        } catch {
            const homeCalorieConsumedEl = document.getElementById('home-calorie-consumed');
            if (homeCalorieConsumedEl) {
                homeCalorieConsumedEl.textContent = '--';
            }
        }
    }

// 消耗卡路里
    async function refreshCaloriesBurnedAndHome(date) {
        const dateStr = date ? date.toISOString().substring(0, 10) : (new Date()).toISOString().substring(0, 10);
        try {
            const res = await fetch(`${API_BASE_URL}/getCaloriesUsedToday?date=${encodeURIComponent(dateStr)}`);
            const data = await res.json();
            const calorieBurnedEl = document.getElementById('calorie-burned');
            if (calorieBurnedEl)
                calorieBurnedEl.textContent = (typeof data.calories === 'number') ? data.calories.toFixed(2) + ' kcal' : '--';
            const homeCalorieBurnedEl = document.getElementById('home-calorie-burned');
            if (homeCalorieBurnedEl)
                homeCalorieBurnedEl.textContent = (typeof data.calories === 'number') ? data.calories.toFixed(2) + ' kcal' : '--';
        } catch {
            const calorieBurnedEl = document.getElementById('calorie-burned');
            if (calorieBurnedEl) calorieBurnedEl.textContent = '--';
            const homeCalorieBurnedEl = document.getElementById('home-calorie-burned');
            if (homeCalorieBurnedEl) homeCalorieBurnedEl.textContent = '--';
        }
    }

    //
    // // 久坐提醒（可自定义或后端接口）
    function refreshHomeSedentary() {
        let now = new Date();
        let hours = now.getHours();
        let value = (hours > 13 && hours < 17) ? 'High' : 'Low';
        let tip = (value === 'High') ? 'Time to get up and move around!' : 'Keep moving for better health!';
        const sedentaryValueEl = document.getElementById('home-sedentary-value');
        if (sedentaryValueEl) sedentaryValueEl.textContent = value;
        const sedentaryTipEl = document.getElementById('home-sedentary-tip');
        if (sedentaryTipEl) sedentaryTipEl.textContent = tip;
    }
    //
    // 统一定时器
    setInterval(refreshStepsAndHome, 2000);
    setInterval(refreshHeartRateAndHome, 2000);
    setInterval(() => refreshHomeWaterToday(), 2000);
    setInterval(() => refreshCaloriesConsumedHome(), 2000);
    setInterval(() => refreshCaloriesBurnedAndHome(), 2000);
    setInterval(refreshHomeSedentary, 900000); // 15分钟

    // 页面初次加载
    refreshStepsAndHome();
    refreshHeartRateAndHome();
    refreshHomeWaterToday();
    refreshCaloriesConsumedHome();
    refreshCaloriesBurnedAndHome();
    refreshHomeSedentary();

    /*/===== =  = = === = ===== ===== =   HOME END   === ====== ============== ======= ===/*/

    async function handleEnterWorkoutMode() {
        // 检查是否有未结束的运动
        try {
            const resp = await fetch(`${API_BASE_URL}/api/workout/realtime`);
            const data = await resp.json();
    
            if (resp.ok && data.active) {
                // 有未结束的运动，直接进入计时UI
                startSection.style.display = 'none';
                statusSection.classList.add('active');
                secondsElapsed = data.duration || 0;
                // 让select和界面同步
                exerciseTypeSelect.value = data.exercise_type;
                updateWorkoutDisplay();
                startTimer();
                startRealtimeDataPolling();
            } else {
                // 没有未结束的运动，显示选择界面
                statusSection.classList.remove('active');
                startSection.style.display = 'flex';
                // 建议这里也reset一下状态，防止显示脏数据
                resetStatusDisplay();
                clearInterval(realtimeDataInterval);
                clearInterval(workoutTimer);
                secondsElapsed = 0;
            }
        } catch (e) {
            // 网络错误等，fallback到选择界面
            statusSection.classList.remove('active');
            startSection.style.display = 'flex';
            resetStatusDisplay();
            clearInterval(realtimeDataInterval);
            clearInterval(workoutTimer);
            secondsElapsed = 0;
        }
    }

    const today = new Date();
    setupWaterHistoryPicker();
    refreshWater(new Date());

    // 日期相关的工具函数
    function formatDateInput(d) {
        return d.toISOString().substring(0,10);
    }
    function addDays(dateStr, offset) {
        const d = new Date(dateStr);
        d.setDate(d.getDate() + offset);
        return formatDateInput(d);
    }
    function isToday(date) {
        const now = new Date();
        return date.getFullYear() === now.getFullYear() &&
               date.getMonth() === now.getMonth() &&
               date.getDate() === now.getDate();
    }

    // 统一处理所有历史日期控件
    function setupHistoryDatePicker(inputId, prevBtnId, nextBtnId, onChange) {
        const input = document.getElementById(inputId);
        const prevBtn = document.getElementById(prevBtnId);
        const nextBtn = document.getElementById(nextBtnId);
        if (!input) return;
        input.value = formatDateInput(new Date());
        prevBtn?.addEventListener('click', () => {
            input.value = addDays(input.value, -1);
            input.dispatchEvent(new Event('change'));
        });
        nextBtn?.addEventListener('click', () => {
            input.value = addDays(input.value, 1);
            input.dispatchEvent(new Event('change'));
        });
        if (typeof onChange === 'function') {
            input.addEventListener('change', () => {
                onChange(new Date(input.value));
            });
        }
    }
    function refreshWater(date) {
        const dateStr = formatDateInput(date || new Date());
        fetch(`${API_BASE_URL}/getWaterAmount?date=${encodeURIComponent(dateStr)}`)
            .then(res => res.json())
            .then(data => {
                document.getElementById('water-today').textContent = data.amount ? (data.amount + 'ml') : '--';
            })
            .catch(() => {
                document.getElementById('water-today').textContent = '--';
            });
    }

    // =============== 心率相关 START ===============
    const hrDateInput = document.getElementById('hr-history-date');
    const hrFromInput = document.getElementById('hr-history-from');
    const hrToInput = document.getElementById('hr-history-to');
    const loadHistoryBtn = document.getElementById('loadHrHistoryBtn');
    const prevBtn = document.getElementById('prevHrDate');
    const nextBtn = document.getElementById('nextHrDate');
    const historyListDiv = document.getElementById('hr-history-list');
    const historyChartCanvas = document.getElementById('heartRateChart');
    let heartRateChart = null;
    let heartRateRealtimeTimer = null;
    let alarmRealtimeTimer = null;
    let isHeartRateRealtime = true;

    // 心率数字始终2秒自动刷新
    async function refreshHeartRate() {
        try {
            const res = await fetch(`${API_BASE_URL}/getHeartRate`);
            const data = await res.json();
            document.getElementById('heart-rate-value').textContent = data.value ?? '--';
        } catch {
            document.getElementById('heart-rate-value').textContent = '--';
        }
    }
    setInterval(refreshHeartRate, 2000);

    // 实时模式：拉取当天00:00:00~23:59:59
    async function refreshHeartRateRealtimeChart() {
        if (!isHeartRateRealtime) return;
        const now = new Date();
        const todayStr = now.toISOString().slice(0, 10);
        const start = new Date(`${todayStr}T00:00:00`);
        const end = new Date(`${todayStr}T23:59:59`);
        try {
            const url = `${API_BASE_URL}/getHeartRateTimeSlot?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`;
            const res = await fetch(url);
            if (!res.ok) throw new Error('HTTP error');
            const data = await res.json();
    
            // 按小时分组所有记录
            const chartDataPoints = [];
            data.records.forEach(item => {
                const t = new Date(item.time);
                const hour = t.getHours();
                chartDataPoints.push({
                    x: hour, // x轴用小时整点
                    y: item.value,
                    abnormal: item.abnormal
                });
            });
    
            updateHeartRateChart(chartDataPoints);
        } catch (e) {}
    }
    function startHeartRateRealtimeChart() {
        isHeartRateRealtime = true;
        refreshHeartRateRealtimeChart();
        if (heartRateRealtimeTimer) clearInterval(heartRateRealtimeTimer);
        heartRateRealtimeTimer = setInterval(refreshHeartRateRealtimeChart, 2000);
    }
    function stopHeartRateRealtimeChart() {
        if (heartRateRealtimeTimer) {
            clearInterval(heartRateRealtimeTimer);
            heartRateRealtimeTimer = null;
        }
    }

    // 进入心率页面时自动实时，离开时停止
    function observeHrSection() {
        const hrSection = document.getElementById('hr-monitor');
        if (!hrSection) return;
        const observer = new MutationObserver(() => {
            if (hrSection.classList.contains('active')) {
                startHeartRateRealtimeChart();
                startAlarmRealtime(); // 新增：进入页面时启动alarm定时器
                // 恢复日期控件到今天
                if (hrDateInput) hrDateInput.value = formatDateInput(new Date());
                if (hrFromInput) hrFromInput.value = "00:00:00";
                if (hrToInput) hrToInput.value = "23:59:59";
            } else {
                stopHeartRateRealtimeChart();
                stopAlarmRealtime(); // 新增：离开页面时关闭alarm定时器
            }
        });
        observer.observe(hrSection, { attributes: true, attributeFilter: ['class'] });
    }
    observeHrSection();

    // 历史模式：拉取指定时间段
    async function refreshHeartRateHistory(start, end) {
        if (!historyListDiv) return;
        historyListDiv.innerHTML = 'Loading...';
        if (historyChartCanvas) {
            const ctx = historyChartCanvas.getContext('2d');
            ctx.clearRect(0, 0, historyChartCanvas.width, historyChartCanvas.height);
        }
        try {
            const url = `${API_BASE_URL}/getHeartRateTimeSlot?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`;
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
            const data = await res.json();
            if (!data || !Array.isArray(data.records) || data.records.length === 0) {
                historyListDiv.innerHTML = '<span style="color:#aaa">No data for this time range.</span>';
                updateHeartRateChart([]); // 只传空数组
                updateAbnormalRecordsDisplay([]);
                return;
            }
            // FIX: 保证有abnormal字段，并x为小时
            const chartDataPoints = data.records.map(item => ({
                x: new Date(item.time).getHours(),
                y: item.value,
                abnormal: item.abnormal
            }));
            updateHeartRateChart(chartDataPoints); // 只传一个参数
    
            const abnormalRecords = data.records.filter(item => item.abnormal);
            updateAbnormalRecordsDisplay(abnormalRecords);
            historyListDiv.innerHTML = '';
        } catch (error) {
            historyListDiv.innerHTML = '<span style="color:red">Failed to load data. Please check server connection.</span>';
            updateHeartRateChart([]);
            updateAbnormalRecordsDisplay([]);
        }
    }
    
    // 日期/时间控件功能，Load History/箭头/输入变化都会触发历史查询，并且进入历史模式
    function setupHrHistoryPicker() {
        function triggerHistory() {
            isHeartRateRealtime = false;
            stopHeartRateRealtimeChart();
            const dateStr = hrDateInput.value;
            const fromStr = hrFromInput.value || "00:00:00";
            const toStr = hrToInput.value || "23:59:59";
            const start = new Date(`${dateStr}T${fromStr}`);
            const end = new Date(`${dateStr}T${toStr}`);
            refreshHeartRateHistory(start, end);
            stopAlarmRealtime();
        }
        // 左右箭头
        prevBtn?.addEventListener('click', () => {
            if (hrDateInput) {
                const oldDate = new Date(hrDateInput.value);
                oldDate.setDate(oldDate.getDate() - 1);
                hrDateInput.value = formatDateInput(oldDate);
                triggerHistory();
            }
        });
        nextBtn?.addEventListener('click', () => {
            if (hrDateInput) {
                const oldDate = new Date(hrDateInput.value);
                oldDate.setDate(oldDate.getDate() + 1);
                hrDateInput.value = formatDateInput(oldDate);
                triggerHistory();
            }
        });
        // 日期和时间输入
        [hrDateInput, hrFromInput, hrToInput].forEach(input => {
            if (input) input.addEventListener('change', triggerHistory);
        });
        // Load History按钮
        if (loadHistoryBtn) {
            loadHistoryBtn.addEventListener('click', triggerHistory);
        }
    }
    setupHrHistoryPicker();

    // 点击其它tab再切回来自动恢复实时
    function formatDateInput(d) {
        return d.toISOString().substring(0, 10);
    }

    // Chart.js渲染
    function updateHeartRateChart(dataPoints) {
        if (typeof Chart === 'undefined') return;
        if (!historyChartCanvas) return;
    
        // 分颜色
        const normalPoints = dataPoints.filter(pt => !pt.abnormal);
        const abnormalPoints = dataPoints.filter(pt => pt.abnormal);
    
        // 按小时分组（x轴就是0~23整数）
        const chart = heartRateChart;
        if (chart) {
            chart.data.datasets[0].data = normalPoints;
            chart.data.datasets[1].data = abnormalPoints;
            chart.update();
            return;
        }
    
        const ctx = historyChartCanvas.getContext('2d');
        heartRateChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Heart Rate',
                        data: normalPoints,
                        backgroundColor: 'rgba(54, 162, 235, 0.8)', // 蓝色
                        pointRadius: 5,
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        showLine: false,
                    },
                    {
                        label: 'Abnormal',
                        data: abnormalPoints,
                        backgroundColor: 'rgba(239, 68, 68, 0.8)', // 红色
                        pointRadius: 7,
                        borderColor: '#ef4444',
                        borderWidth: 2,
                        showLine: false,
                    }
                ]
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: 'Heart Rate Scatter Plot (Today)',
                        color: '#222',
                        font: { size: 20, weight: 'bold' },
                        padding: { top: 16, bottom: 10 }
                    },
                    legend: {
                        display: true,
                        labels: {
                            color: '#222',
                            font: { size: 15 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Hour: ${context.parsed.x}, Heart Rate: ${context.parsed.y} bpm`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 23,
                        title: {
                            display: true,
                            text: 'Hour',
                            color: '#222',
                            font: { size: 16, weight: 'bold' }
                        },
                        ticks: {
                            stepSize: 1,
                            color: '#222',
                            font: { size: 13 },
                            callback: value => value // 只显示整数小时
                        },
                        grid: { color: '#e0e7ef' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Heart Rate (BPM)',
                            color: '#222',
                            font: { size: 16, weight: 'bold' }
                        },
                        beginAtZero: true,
                        grid: { color: '#f3f4f6' },
                        ticks: { color: '#222', font: { size: 13 } }
                    }
                },
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    // 报警模块
    function startAlarmRealtime() {
        refreshAlarmRealtime();
        if (alarmRealtimeTimer) clearInterval(alarmRealtimeTimer);
        alarmRealtimeTimer = setInterval(refreshAlarmRealtime, 2000);
    }
    function stopAlarmRealtime() {
        if (alarmRealtimeTimer) {
            clearInterval(alarmRealtimeTimer);
            alarmRealtimeTimer = null;
        }
    }
    async function refreshAlarmRealtime() {
        // 拉取今天00:00~23:59所有心率，筛选abnormal
        const now = new Date();
        const todayStr = now.toISOString().slice(0, 10);
        const start = new Date(`${todayStr}T00:00:00`);
        const end = new Date(`${todayStr}T23:59:59`);
        try {
            const url = `${API_BASE_URL}/getHeartRateTimeSlot?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`;
            const res = await fetch(url);
            if (!res.ok) return;
            const data = await res.json();
            const abnormalRecords = (data.records || []).filter(item => item.abnormal);
            updateAbnormalRecordsDisplay(abnormalRecords);
        } catch {}
    }
    function updateAbnormalRecordsDisplay(records) {
        const container = document.getElementById('abnormal-records-container');
        if (!container) return;
        if (records.length === 0) {
            container.innerHTML = `
                <div class="alarm-no-abnormal">
                    <span class="alarm-no-abnormal-icon">✔️</span>
                    <span class="alarm-no-abnormal-text">There is no abnormal record on this day.</span>
                </div>
            `;
            container.classList.remove("alarm-has-abnormal");
            container.classList.add("alarm-no-abnormal-bg");
            return;
        }
        const listItems = records.map(item => {
            const t = new Date(item.time);
            const timeStr = t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            return `
                <div class="abnormal-record-item">
                    <span class="record-time">${timeStr}</span>
                    <span class="record-value">${item.value ?? '--'} bpm</span>
                </div>
            `;
        }).join('');
        container.innerHTML = `
            <div class="abnormal-list-header">
                ⚠️ Abnormal Records
            </div>
            ${listItems}
        `;
        container.classList.remove("alarm-no-abnormal-bg");
        container.classList.add("alarm-has-abnormal");
    }

    // =============== 心率相关 END ===============
        // 页面加载时自动拉取 calorie target 并填入
    async function fetchCalorieTarget() {
        try {
            const res = await fetch(`${API_BASE_URL}/getCaloriesTarget`);
            if (res.ok) {
                const data = await res.json();
                const val = data.target;
                const input = document.getElementById('calorieTargetInput');
                if (input && val) input.value = val;
            }
        } catch {}
    }
    fetchCalorieTarget();
        // 页面加载时自动拉取 step target 并填入
        async function fetchStepTarget() {
            try {
                const res = await fetch(`${API_BASE_URL}/getStepTarget`);
                if (res.ok) {
                    const data = await res.json();
                    const val = data.target;
                    const input = document.getElementById('stepTargetInput');
                    if (input && val) input.value = val;
                }
            } catch {}
        }
        fetchStepTarget();
fetchCalorieTarget();
    // 3. 饮食建议（卡路里）历史
    setupHistoryDatePicker('diet-date-picker', 'prevDietDate', 'nextDietDate', (date) => {
        refreshDietSuggestion(date);
        updateDietSettingsVisibility(date);
    });

    // ================== 步数 ==================
    document.getElementById('setStepTargetBtn')?.addEventListener('click', async function(e) {
        e.preventDefault();
        const val = document.getElementById('stepTargetInput').value;
        if (val && parseInt(val) > 0) {
            try {
                const resp = await fetch(`${API_BASE_URL}/setStepTarget`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: parseInt(val) })
                });
                if (resp.ok) {
                    alert('Steps target set to ' + val + ' steps!');
                } else {
                    const errMsg = await resp.text();
                    alert('Failed to set target! Server returned: ' + resp.status + ' ' + errMsg);
                }
            } catch (err) {
                alert('Failed to set step target! ' + err);
            }
        } else {
            alert('Please enter a valid step target!');
        }
    });

    let latestSteps = 0; // 保存最新步数
    // 获取步数
    async function refreshSteps() {
        try {
            const res = await fetch(`${API_BASE_URL}/getSteps`);
            const data = await res.json();
            const steps = Number(data.steps) || 0;
            document.getElementById('step-count').textContent = data.steps ?? '--';
            latestSteps = steps;
            // calculateCaloriesBurned();
        } catch {
            document.getElementById('step-count').textContent = '--';
        }
    }

    refreshSteps();
    setInterval(refreshSteps, 2000); // 每两秒自动刷新


        // ================== Routine 地图轨迹显示 ==================
    let routineMap = null;
    let routinePolyline = null;
    let routineMarkers = [];
    // 初始化地图
    function initRoutineMap() {
    const mapDiv = document.getElementById('routine-map');
    if (!mapDiv) return;

    // 避免尺寸未初始化时创建地图
    if (!mapDiv.offsetWidth || !mapDiv.offsetHeight) {
        // 容器尚未可见，稍后重试
        setTimeout(initRoutineMap, 300);
        return;
    }

    if (routineMap) {
        routineMap.eachLayer(layer => {
            if (layer instanceof L.Polyline || layer instanceof L.Marker) routineMap.removeLayer(layer);
        });
        if (routinePolyline) routinePolyline = null;
        routineMarkers = [];
        return;
    }

    // 初始化地图（使用更现代的底图）
    routineMap = L.map('routine-map', {
        center: [1.2966, 103.7764], // NUS Kent Ridge
        zoom: 15,
        zoomControl: true,
        attributionControl: true
    });


    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> contributors',
        maxZoom: 19
    }).addTo(routineMap);
    }


    // 渲染轨迹
    function renderRoutinePath(points) {
    renderRoutinePathWithSpeed(points);
}

function renderRoutinePathWithSpeed(points) {
    if (!routineMap) return;
    // 清除旧图层
    if (routinePolyline) {
        routineMap.removeLayer(routinePolyline);
        routinePolyline = null;
    }
    routineMarkers.forEach(m => routineMap.removeLayer(m));
    routineMarkers = [];

    if (!points || points.length === 0) {
        document.getElementById('routine-map-msg').textContent = '暂无轨迹数据';
        return;
    }
    document.getElementById('routine-map-msg').textContent = '';

    // 分段画虚线：每段是两个点，根据速度设色
    let polylines = [];
    for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i], p2 = points[i+1];
        const speed = p2.speed ?? p1.speed ?? 0;
        let color;
        if (speed >= 7) color = '#22c55e';      // 绿色，快跑
        else if (speed <= 5) color = '#ef4444'; // 红色，慢走
        else color = '#f59e42';                 // 橙色，中速

        const seg = L.polyline([[p1.lat, p1.lng], [p2.lat, p2.lng]], {
            color: color,
            weight: 2,
            opacity: 0.95,
            lineCap: 'round',
            lineJoin: 'round',
            dashArray: '6,6'         // ★虚线风格
        }).addTo(routineMap);
        polylines.push(seg);
    }

    // 起点终点
    const startIcon = L.icon({ iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png', iconSize: [25, 41], iconAnchor: [12, 41] });
    const endIcon = L.icon({ iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-red.png', iconSize: [25, 41], iconAnchor: [12, 41] });
    const startMarker = L.marker([points[0].lat, points[0].lng], {icon: startIcon, title:'Start'}).addTo(routineMap);
    const endMarker = L.marker([points[points.length-1].lat, points[points.length-1].lng], {icon: endIcon, title:'End'}).addTo(routineMap);
    routineMarkers.push(startMarker, endMarker);

    // 缩放视图
    const allCoords = points.map(pt => [pt.lat, pt.lng]);
    const bounds = L.latLngBounds(allCoords);
    routineMap.fitBounds(bounds, {padding: [30,30]});
    setTimeout(() => routineMap.invalidateSize(), 80);
}

    // 获取当天Routine轨迹
    async function refreshRoutinePath() {
        initRoutineMap();
        const msgBox = document.getElementById('routine-map-msg');
        msgBox.textContent = 'Loading...';
        try {
            // 获取当天时间段
            const now = new Date();
            const start = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
            const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59, 999);
            // 调用后端API获取轨迹
            const resp = await fetch(`${API_BASE_URL}/getRoutine?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`);
            if (!resp.ok) throw new Error('Network error');
            const data = await resp.json();
            // 数据格式要求：[{lat: xx, lng: xx, time: ...}, ...]
            if (!data || !Array.isArray(data.route) || data.route.length === 0) {
                msgBox.textContent = 'No routine data today';
                renderRoutinePath([]);
                return;
            }
            // 转换为坐标数组
            renderRoutinePathWithSpeed(data.route);
        } catch (err) {
            msgBox.textContent = 'Load failure,please check your internet and devices.';
            renderRoutinePath([]);
        }
    }

    // 当Steps页面显示时自动刷新轨迹
    function observeStepsSection() {
        const stepsSection = document.getElementById('steps');
        if (!stepsSection) return;
        const observer = new MutationObserver(() => {
            if (stepsSection.classList.contains('active')) {
                setTimeout(refreshRoutinePath, 350);
            }
        });
        observer.observe(stepsSection, {attributes: true, attributeFilter: ['class']});
    }
    observeStepsSection();

    // 页面初次加载时也尝试加载Routine
    setTimeout(refreshRoutinePath, 600);
    setInterval(refreshRoutinePath, 30000); // 每30秒刷新一次轨迹


    // ================== 水 ==================
    document.getElementById('setWaterTargetBtn')?.addEventListener('click', async function(e) {
    e.preventDefault();
    const val = document.getElementById('waterTargetInput').value;
    if (val && parseInt(val) > 0) {
        try {
            const resp = await fetch(`${API_BASE_URL}/setWaterAmountTarget`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target: parseInt(val) })
            });
            if (resp.ok) {
                alert('Water target set to ' + val + ' ml!');
                // 关键：刷新建议，传当前选中的日期
                const input = document.getElementById('water-history-date');
                const selectedDate = input ? new Date(input.value) : new Date();
                refreshHydrationSuggestion(selectedDate);
            } else {
                const errMsg = await resp.text();
                alert('Failed to set water target! Server returned: ' + resp.status + ' ' + errMsg);
            }
        } catch (err) {
            alert('Failed to set water target! ' + err);
        }
    } else {
        alert('Please enter a valid water target!');
    }
});
    // 获取喝水量
    async function refreshWater() {
        try {
            const res = await fetch(`${API_BASE_URL}/getWaterAmount`);
            const data = await res.json();
            document.getElementById('water-today').textContent = data.amount ? (data.amount + 'ml') : '--';
        } catch {
            document.getElementById('water-today').textContent = '--';
        }
    }
    refreshWater();
    setInterval(refreshWater, 180000); // 每3分钟刷新


    //==================心率数字显示==================
    async function refreshHeartRate() {
        try {
            const res = await fetch(`${API_BASE_URL}/getHeartRate`);
            const data = await res.json();
            document.getElementById('heart-rate-value').textContent = data.value ?? '--';
            // setAlarmStatus(data.abnormal);
        } catch {
            document.getElementById('heart-rate-value').textContent = '--';
            // setAlarmStatus(false);
        }
    }
    setInterval(refreshHeartRate, 2000); //


    // 报警区样式切换

    // ====== Chatting 子页面导航 ======
    // 1. 在侧边栏添加 Chatting 导航项（假设HTML已添加 .nav-item[data-content="chatting"]）
    document.querySelectorAll('.nav-item[data-content="chatting"], .nav-header[data-content="chatting"], .submenu li[data-content="chatting"]').forEach(item => {
        item.addEventListener('click', function () {
            // 隐藏所有内容区
            document.querySelectorAll('.content-section, .welcome-message').forEach(sec => sec.classList.remove('active'));
            // 显示聊天区
            const chatSection = document.getElementById('chatting');
            if (chatSection) chatSection.classList.add('active');
            // // 设置标题
            // const contentTitle = document.getElementById('content-title');
            // if (contentTitle) contentTitle.textContent = 'Chatting (GPT)';
        });
    });

    // 2. 聊天区逻辑（独立于settings弹窗）
    (function() {
        const input = document.getElementById('chatting-input');
        const sendBtn = document.getElementById('chatting-send-btn');
        const chatBody = document.getElementById('chatting-body');

        if (!input || !sendBtn || !chatBody) return;

        function appendMsg(msg, sender) {
            const div = document.createElement('div');
            div.className = 'chat-message ' + sender;
            if (sender === 'bot' && window.marked) {
                div.innerHTML = window.marked.parse(msg);
            } else {
                div.textContent = msg;
            }
            chatBody.appendChild(div);
            chatBody.scrollTop = chatBody.scrollHeight;
        }

        async function query(data) {
            const response = await fetch(
                "http://192.168.137.14:3000/api/v1/prediction/bbdbceeb-0261-4118-9a40-d3fd44f32edc",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                }
            );
            const result = await response.json();
            return result;
        }

        async function sendMessage() {
            const userMsg = input.value.trim();
            if (!userMsg) return;
            appendMsg(userMsg, 'user');
            input.value = '';
            const thinkingDiv = document.createElement('div');
            thinkingDiv.className = 'chat-message bot';
            thinkingDiv.textContent = "Thinking...";
            chatBody.appendChild(thinkingDiv);
            try {
                const data = await query({ "question": userMsg });
                thinkingDiv.remove();
                const botResponse = data.text ?? data.response ?? data.answer ?? JSON.stringify(data);
                appendMsg(botResponse, 'bot');
            } catch (error) {
                thinkingDiv.remove();
                appendMsg("Connection error. Please try again.", 'bot');
            }
        }

        sendBtn.onclick = sendMessage;
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    })();



// ====== Meal Photo Recognition (餐食照片识别) ======

    function updateDietSettingsVisibility(date) {
        const setWeightCard = document.getElementById('set-weight-card');
        const calorieTargetCard = document.getElementById('calorie-target-card');
        // 只在当天显示
        if (isToday(date)) {
            setWeightCard.style.display = '';
            calorieTargetCard.style.display = '';
        } else {
            setWeightCard.style.display = 'none';
            calorieTargetCard.style.display = 'none';
        }
    }
    // 记录年龄（本地存储，防刷新丢失）
    let userAge = Number(localStorage.getItem('userAge')) || 20; // 默认20岁

    // 初始化年龄输入框
    const ageInput = document.getElementById('ageInput');
    if (ageInput) {
        ageInput.value = userAge;
        document.getElementById('setAgeBtn').addEventListener('click', function() {
            const val = Number(ageInput.value);
            if (val > 0) {
                userAge = val;
                localStorage.setItem('userAge', userAge);
                // 如果有需要，可以在这里调用和年龄相关的计算
            } else {
                alert('Please enter a valid age!');
            }
        });
    }
    document.getElementById('setAgeBtn').addEventListener('click', async function() {
        const val = Number(ageInput.value);
        if (val > 0) {
            userAge = val;
            localStorage.setItem('userAge', userAge);
            try {
                const resp = await fetch(`${API_BASE_URL}/setAge`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ age: val })
                });
                if (resp.ok) {
                    alert('Age set to ' + val + '!');
                } else {
                    const errMsg = await resp.text();
                    alert('Failed to set age! Server returned: ' + resp.status + ' ' + errMsg);
                }
            } catch (err) {
                alert('Failed to connect to backend!');
            }
        } else {
            alert('Please enter a valid age!');
        }
    });

    // ========== 性别选择与设置 ==========
    let userGender = localStorage.getItem('userGender') || "male";

    // 渲染性别选中状态
    function renderGender() {
        document.getElementById('circle-male')?.classList.toggle('selected', userGender === "male");
        document.getElementById('circle-female')?.classList.toggle('selected', userGender === "female");
    }
    renderGender();

    // 点击行切换性别
    document.querySelectorAll('.gender-row').forEach(row => {
        row.addEventListener('click', function() {
            userGender = this.getAttribute('data-gender');
            renderGender();
        });
    });

    // 页面加载时从后端读取（如果有设置）
    async function fetchGenderFromServer() {
        try {
            const res = await fetch(`${API_BASE_URL}/getGender`);
            if (res.ok) {
                const data = await res.json();
                if (data.gender === "male" || data.gender === "female") {
                    userGender = data.gender;
                    localStorage.setItem('userGender', userGender);
                    renderGender();
                }
            }
        } catch {}
    }
    fetchGenderFromServer();

    // 点击Set按钮，保存到后端和本地
    document.getElementById('setGenderBtn').addEventListener('click', async function() {
            if (!userGender) {
                alert('Please select a gender!');
                return;
            }
            localStorage.setItem('userGender', userGender);
            try {
                const resp = await fetch(`${API_BASE_URL}/setGender`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ gender: userGender })
                });
                if (resp.ok) {
                    alert('Gender set to ' + userGender + '!');
                } else {
                    alert('Failed to set gender!');
                }
            } catch {
                alert('Failed to connect to backend!');
            }
        });
        
        // ====== Diet Suggestion (日历和数据联动) ======
        async function refreshDietSuggestion(date) {
            if (!date) date = new Date();
            // 先拉取消耗和摄入
            let consumed = 0, used = 0, target = 2000;
            try {
                const res1 = await fetch(`${API_BASE_URL}/getCaloriesUsedToday?date=${encodeURIComponent(date.toISOString())}`);
                const data1 = await res1.json();
                used = Number(data1.calories) || 0;
                document.getElementById('calorie-burned').textContent = used.toFixed(2) + ' kcal';
            } catch {
                document.getElementById('calorie-burned').textContent = '--';
            }
            try {
                const res2 = await fetch(`${API_BASE_URL}/getCaloriesConsumed?date=${encodeURIComponent(date.toISOString())}`);
                const data2 = await res2.json();
                consumed = Number(data2.calories) || 0;
                document.getElementById('calorie-consumed').textContent = consumed + 'kcal';
            } catch {
                document.getElementById('calorie-consumed').textContent = '--';
            }
            // 3. 获取目标
            try {
                const res3 = await fetch(`${API_BASE_URL}/getCaloriesTarget?date=${encodeURIComponent(date.toISOString())}`);
                const data3 = await res3.json();
                if (typeof data3.target === 'number') target = data3.target;
            } catch {
                // target 保持默认2000
            }
            const net = used - consumed;
            // 生成建议
            let suggestion = "Eat a balanced diet of protein, vegetables, fruit, and whole grains.";
            // 这一步建议你可以结合你的卡路里目标

            if (Math.abs(net - target) <= 100) {
                suggestion = "Great! Your net calorie is close to your daily goal.";
            } else if (net > target + 100) {
                suggestion = "You've burned significantly more than your target. Consider increasing food intake or reducing activity.";
            } else if (net < target - 100) {
                suggestion = "Your net calorie is lower than your daily goal. Consider lighter meals or more exercise.";
            }
            // 辅助：如果摄入远大于消耗，也给一个警告
            if (consumed > used + 300) {
                suggestion = "Your calorie intake is much higher than consumption. Consider more exercise or lighter meals.";
            } else if (used > consumed + 300) {
                suggestion = "You've burned much more than you consumed. Ensure you eat enough for recovery.";
            }
            document.getElementById('diet-suggestion-text').textContent = suggestion;
        }

    async function refreshCaloriesUsedToday(date) {
        try {
            const res = await fetch(`${API_BASE_URL}/getCaloriesUsedToday?date=${encodeURIComponent(date.toISOString())}`);
            const data = await res.json();
            document.getElementById('calorie-burned').textContent = data.calories ?? '--';
        } catch {
            document.getElementById('calorie-burned').textContent = '--';
        }
    }
    async function refreshCaloriesGainedToday(date) {
        try {
            const res = await fetch(`${API_BASE_URL}/getCaloriesConsumed?date=${encodeURIComponent(date.toISOString())}`);
            const data = await res.json();
            document.getElementById('calorie-consumed').textContent = data.calories + 'kcal'?? '--';
        } catch {
            document.getElementById('calorie-consumed').textContent = '--';
        }
    }


    // 页面初始加载刷新饮食建议
    refreshDietSuggestion(new Date());

    //设定卡路里目标
    document.getElementById('setCalorieTargetBtn')?.addEventListener('click', async function(e) {
        e.preventDefault();
        const val = document.getElementById('calorieTargetInput').value;
        if (val && parseInt(val) > 0) {
            try {
                const resp = await fetch(`${API_BASE_URL}/setCaloriesTarget`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: parseInt(val) })
                });
                if (resp.ok) {
                    alert('Calorie target set to ' + val + ' kcal!');
                    // 延迟0.5秒再刷新
                    setTimeout(() => {
                        const dietDateInput = document.getElementById('diet-date-picker');
                        let selectedDate = new Date();
                        if (dietDateInput && dietDateInput.value) {
                            selectedDate = new Date(dietDateInput.value);
                        }
                        refreshDietSuggestion(selectedDate);
                    }, 500);
                }else {
                    const errMsg = await resp.text();
                    alert('Failed to set calorie target! Server returned: ' + resp.status + ' ' + errMsg);
                }
            } catch (err) {
                alert('Failed to set calorie target! ' + err);
            }
        } else {
            alert('Please enter a valid calorie target!');
        }
    });


// ========== 悬浮按钮与竖型对话框联动 ==========
(function() {
    const fab = document.getElementById('settings-chat-btn');
    const dialog = document.getElementById('settings-chat-dialog');
    const closeBtn = document.getElementById('settings-chat-close-btn');
    const input = document.getElementById('settings-chat-input');
    const sendBtn = document.getElementById('settings-chat-send-btn');
    const chatBody = document.getElementById('settings-chat-body');

    // 确保对话框初始隐藏
    dialog.classList.remove('active');
    dialog.style.display = 'none';

    // // 按钮点击弹出聊天框
    // fab.onclick = function() {
    //     dialog.classList.add('active');
    //     dialog.style.display = 'flex';
    //     setTimeout(() => input.focus(), 180);
    // };

    // // 关闭按钮关闭聊天框
    // closeBtn.onclick = function() {
    //     dialog.classList.remove('active');
    //     dialog.style.display = 'none';
    // };

    // 添加消息到聊天框
    function appendMsg(msg, sender) {
        const div = document.createElement('div');
        div.className = 'chat-message ' + sender;
        if (sender === 'bot' && window.marked) {
            // 对机器人消息使用 Markdown 解析
            div.innerHTML = window.marked.parse(msg);
        } else {
            // 用户消息或未加载 marked.js 时，作为纯文本处理
            div.textContent = msg;
        }
        chatBody.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    // 新增：统一query函数
    async function query(data) {
        const response = await fetch(
            "http://192.168.137.85:3000/api/v1/prediction/bbdbceeb-0261-4118-9a40-d3fd44f32edc",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }
        );
        const result = await response.json();
        return result;
    }

    async function sendMessage() {
        const userMsg = input.value.trim();
        if (!userMsg) return;

        appendMsg(userMsg, 'user');
        input.value = '';

        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'chat-message bot';
        thinkingDiv.textContent = "Thinking...";
        chatBody.appendChild(thinkingDiv);

        try {
            // 使用query方法
            const data = await query({ "question": userMsg });
            thinkingDiv.remove();
            // 优先解析 data.text 字段，如果不存在则兼容旧结构
            const botResponse = data.text ?? data.response ?? data.answer ?? JSON.stringify(data);
            appendMsg(botResponse, 'bot');
        } catch (error) {
            thinkingDiv.remove();
            appendMsg("Connection error. Please try again.", 'bot');
        }
    }

    // 点击聊天框外区域关闭
    document.addEventListener('click', function(e) {
        if (dialog.classList.contains('active')) {
            if (!dialog.contains(e.target) && !fab.contains(e.target)) {
                dialog.classList.remove('active');
                dialog.style.display = 'none';
            }
        }
    });
    // 绑定发送按钮和回车键
    sendBtn.onclick = sendMessage;
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') sendMessage();
    });
})();


// 记录体重（本地存储，防刷新丢失）
let userWeight = Number(localStorage.getItem('userWeight')) || 60; // 默认60kg

// 初始化体重输入框
const weightInput = document.getElementById('weightInput');
if (weightInput) {
    weightInput.value = userWeight;
    document.getElementById('setWeightBtn').addEventListener('click', async function() {
        const val = Number(weightInput.value);
        if (val > 0) {
            try {
                // 发到后端
                const resp = await fetch(`${API_BASE_URL}/setWeight`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ weight: val })
                });
                if (resp.ok) {
                    userWeight = val;
                    localStorage.setItem('userWeight', userWeight);
                    alert('Weight set to ' + val + ' kg!');
                    // calculateCaloriesBurned(); // 设置后立即刷新
                }
                // else {
                //     const errMsg = await resp.text();
                //     alert('Failed to set weight! Server returned: ' + resp.status + ' ' + errMsg);
                // }
            } catch (err) {
                alert('Failed to set weight! ' + err);
            }
        } else {
            alert('Please enter a valid weight!');
        }
    });
}

// // 页面加载时自动拉取后端体重
// async function fetchUserWeight() {
//     try {
//         const resp = await fetch(`${API_BASE_URL}/getWeight`);
//         if (resp.ok) {
//             const data = await resp.json();
//             if (typeof data.weight === 'number' && !isNaN(data.weight)) {
//                 userWeight = data.weight;
//                 weightInput.value = userWeight;
//                 localStorage.setItem('userWeight', userWeight);
//             }
//         }
//     } catch {}
// }
// if (weightInput) fetchUserWeight();

// Nutrition Management 日期组件
function refreshNutritionDatePicker(date) {
    nutritionDate = date;
    document.getElementById('nutrition-date-picker').value = formatDateInput(date);
    fetchNutritionData(date);
}
// 添加 Check 按钮逻辑
function setupNutritionCheckBtn() {
    const checkBtn = document.getElementById('nutritionCheckBtn');
    if (!checkBtn) return;
    checkBtn.addEventListener('click', () => {
        renderAllPhotoLinks(); // 只刷新对应日期的食物链接
    });
}

// Nutrition Management 获取目标和摄入量
async function fetchNutritionData(date) {
    try {
        // 获取目标
        const targetRes = await fetch(`${API_BASE_URL}/getCaloriesTarget?date=${encodeURIComponent(date.toISOString())}`);
        const targetData = await targetRes.json();
        nutritionTarget = targetData.target ?? 2000;

        // 获取摄入量
        const consumedRes = await fetch(`${API_BASE_URL}/getCaloriesConsumed?date=${encodeURIComponent(date.toISOString())}`);
        const consumedData = await consumedRes.json();
        nutritionConsumed = consumedData.calories ?? 0;
    } catch {
        nutritionTarget = 2000; nutritionConsumed = 0;
    }
}


// Nutrition Management New页面跳转逻辑
document.getElementById('nutritionNewBtn').addEventListener('click', function() {
    document.getElementById('Nutrition Management').classList.remove('active');
    document.getElementById('nutrition-new').classList.add('active');
    renderNutritionNewPage();
});


// Nutrition New页面图片和餐段逻辑
function renderNutritionNewPage() {
    document.getElementById('nutritionNewPhotoPreview').style.display = 'none';
    document.getElementById('nutritionNewPhotoText').style.display = 'block';
    document.getElementById('nutritionUploadStatus').textContent = '';

    let hour = new Date().getHours();
    let defaultType = 'Snack';
    if (hour >= 6 && hour < 11) defaultType = 'Breakfast';
    else if (hour >= 11 && hour < 15) defaultType = 'Lunch';
    else if (hour >= 15 && hour < 24) defaultType = 'Dinner';

    document.getElementById('nutritionMealTypeBtn').textContent = defaultType;
    document.getElementById('nutritionMealTypeBtn').dataset.type = defaultType;
    selectedMealType = defaultType;
    document.getElementById('nutritionMealTypeDropdown').style.display = 'none';
}


// 餐段下拉弹出
document.getElementById('nutritionMealTypeBtn').addEventListener('click', function() {
    const dropdown = document.getElementById('nutritionMealTypeDropdown');
    dropdown.classList.toggle('active');
});

document.querySelectorAll('#nutritionMealTypeDropdown div').forEach(item => {
    item.addEventListener('click', function() {
        document.getElementById('nutritionMealTypeBtn').textContent = this.textContent;
        selectedMealType = this.dataset.type; // 记录当前选中的餐段
        const dropdown = document.getElementById('nutritionMealTypeDropdown');
        dropdown.classList.remove('active'); // 统一用active类收起
    });
});

document.querySelectorAll('#nutritionMealTypeDropdown div').forEach(item => {
    item.addEventListener('click', function() {
        document.getElementById('nutritionMealTypeBtn').textContent = this.textContent;
        document.getElementById('nutritionMealTypeBtn').dataset.type = this.dataset.type;
        document.getElementById('nutritionMealTypeDropdown').style.display = 'none';
    });
});

// 图片选择与上传逻辑（与原按钮一致，只是ID不同）
let nutritionNewPhotoBase64 = "";
document.getElementById('nutritionChooseFileBtn').onclick = function() {
    document.getElementById('nutritionMealPhotoInput').click();
};
document.getElementById('nutritionMealPhotoInput').onchange = function(e) {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(ev) {
            const img = document.getElementById('nutritionNewPhotoPreview');
            img.src = ev.target.result;
            img.style.display = 'block';
            document.getElementById('nutritionNewPhotoText').style.display = 'none';
            nutritionNewPhotoBase64 = ev.target.result;
        };
        reader.readAsDataURL(file);
    }
};

// 1. 记录当前选择的餐段
let selectedMealType = 'Breakfast'; // 默认早餐
const pendingPhotos = []; // 临时保存本次 new 上传的照片及餐段

// 选择餐段时更新
document.querySelectorAll('#nutritionMealTypeDropdown div').forEach(item => {
    item.addEventListener('click', function() {
        document.getElementById('nutritionMealTypeBtn').textContent = this.textContent;
        selectedMealType = this.dataset.type; // 记录当前选中的餐段
        document.getElementById('nutritionMealTypeDropdown').style.display = 'none';
    });
});

// 2. 上传图片后，把结果和餐段存进 pendingPhotos
// 上传后渲染新链接也传日期
document.getElementById('nutritionUploadBtn').onclick = async function() {
    const uploadStatus = document.getElementById('nutritionUploadStatus');
    uploadStatus.textContent = 'Uploading...';
    if (!nutritionNewPhotoBase64) {
        uploadStatus.textContent = 'Please choose a photo first!';
        return;
    }
    try {
        const resp1 = await fetch(`${API_BASE_URL}/updateFoodPhoto`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                photo: nutritionNewPhotoBase64,
                mealType: selectedMealType,
                photoDate: nutritionDate ? formatDateInput(nutritionDate) : formatDateInput(new Date())
            })
        });
        const photoRes = await resp1.json();
        uploadStatus.textContent = photoRes.msg || 'Upload success!';
        if (photoRes.photo_id) {
            // 上传成功，解禁 Measure Weight 按钮
            document.getElementById('nutritionNewMeasureBtn').disabled = false;
            let filename = photoRes.filename;
            if (!filename && photoRes.file_path) {
                filename = photoRes.file_path.split('/').pop();
            }
            addPhotoLinkToMeal(selectedMealType, filename, photoRes.photo_id, "Unnamed photo", formatDateInput(nutritionDate));
        }
    } catch (err) {
        uploadStatus.textContent = 'Upload or recognition failed!';
    }
};


// 主页面所有链接都用 photoId 唯一定位，onclick调用 showPhotoDetailPageById
function addPhotoLinkToMeal(mealType, filename, photoId, displayName, photoDate) {
    const mealLinksId = {
        'Breakfast': 'breakfastLinks',
        'Lunch': 'lunchLinks',
        'Dinner': 'dinnerLinks',
        'Snack': 'snackLinks'
    }[mealType];
    if (!mealLinksId) return;
    const linksDiv = document.getElementById(mealLinksId);
    if ([...linksDiv.children].some(link =>
        link.dataset && (link.dataset.photoId == photoId))) {
        return;
    }
    const link = document.createElement('a');
    link.href = '#';
    link.className = 'nutrition-meal-link';
    link.textContent = displayName || filename;
    link.dataset.photoId = photoId;
    link.onclick = function() {
        showPhotoDetailPageById(photoId, mealType, photoDate);
        return false;
    };
    linksDiv.appendChild(link);
}

document.getElementById('nutritionBackBtn').addEventListener('click', function() {
    document.getElementById('nutrition-new').classList.remove('active');
    document.getElementById('Nutrition Management').classList.add('active');
    // 先拉取数据库的所有照片
    renderAllPhotoLinks().then(() => {
        // 再把本地 pendingPhotos 立即渲染到对应餐段区块（只渲染属于当前日期的）
        pendingPhotos.forEach(photo => {
            // 只渲染当前 nutritionDate 对应的
            if (photo.photoDate === formatDateInput(nutritionDate)) {
                addPhotoLinkToMeal(photo.mealType, photo.filename, photo.photoId, "Unnamed photo");
            }
        });
        // 清空 session
        pendingPhotos.length = 0;
    });
});

// 拉取数据库渲染所有链接，textContent 应该是 food_name || "Unnamed photo"
async function renderAllPhotoLinks() {
    try {
        const res = await fetch(`${API_BASE_URL}/getFoodPhotos`);
        const data = await res.json();
        ['breakfastLinks', 'lunchLinks', 'dinnerLinks', 'snackLinks'].forEach(id => {
            document.getElementById(id).innerHTML = '';
        });
        if (data.photos && Array.isArray(data.photos)) {
            data.photos.filter(photo =>
                photo.photo_date === formatDateInput(nutritionDate)
            ).forEach(photo => {
                const mealType = photo.description || 'Breakfast';
                const photoDate = photo.photo_date || formatDateInput(nutritionDate);
                const linksId = {
                    'Breakfast': 'breakfastLinks',
                    'Lunch': 'lunchLinks',
                    'Dinner': 'dinnerLinks',
                    'Snack': 'snackLinks'
                }[mealType] || 'breakfastLinks';
                const linksDiv = document.getElementById(linksId);
                if ([...linksDiv.children].some(link =>
                    (link.dataset && (link.dataset.photoId == photo.id))
                )) {
                    return;
                }
                const displayName = photo.food_name && photo.food_name !== "??" ? photo.food_name : "Unnamed photo";
                // 传递 photoDate
                addPhotoLinkToMeal(mealType, photo.filename, photo.id, displayName, photoDate);
            });
        }
    } catch (err) {
        document.getElementById('breakfastLinks').innerHTML = '';
    }
    return Promise.resolve();
}



function showPhotoDetailPageById(photoId, mealType, photoDate) {
    document.querySelectorAll('.content-section, .welcome-message').forEach(sec => sec.classList.remove('active'));
    const detailDiv = document.getElementById('photo-detail');
    detailDiv.classList.add('active');
    detailDiv.innerHTML = 'Loading...';

    fetch(`${API_BASE_URL}/getFoodPhotoDetail?photo_id=${photoId}`)
        .then(res => res.json())
        .then(data => {
            const calorie = data.Calorie || '';
            const protein = data.Protein || '';
            const fat = data.Fat || '';
            const carbohydrate = data.Carbohydrate || '';
            const fiber = data.Fiber || '';
            const weight = (data.Weight !== undefined && data.Weight !== null) ? data.Weight : ''; // 真正的 weight
            let foodName = data.food_name || '';
            let dateLabel = photoDate || formatDateInput(nutritionDate);
            let mealTypeLabel = mealType || "Meal";

            detailDiv.innerHTML = `
                <div style="font-size:1.21rem;font-weight:bold;margin-bottom:11px;color:#4e73df;">
                  ${dateLabel} ${mealTypeLabel}
                </div>
                <div style="display:flex; align-items:flex-start;">
                  <img src="${API_BASE_URL}/food_photo_by_filename?filename=${encodeURIComponent(data.file_path.split('/').pop())}&raw=1" style="max-width:300px; max-height:220px; border-radius:10px; margin-right:32px;">
                  <div style="flex:1;">
                    <table style="font-size:1.1em; width:auto;" id="food-info-table">
                      <tr><td style="font-weight:bold;">Food Name</td><td id="food-name-label">${foodName}</td></tr>
                      <tr><td style="font-weight:bold;">Weight(g)</td><td id="food-weight-label">${weight}</td></tr>
                      <tr><td style="font-weight:bold;">Calorie(kcal/100g)</td><td>${calorie}</td></tr>
                      <tr><td style="font-weight:bold;">Protein(/100g)</td><td>${protein}</td></tr>
                      <tr><td style="font-weight:bold;">Fat(/100g)</td><td>${fat}</td></tr>
                      <tr><td style="font-weight:bold;">Carbohydrate(/100g)    </td><td>${carbohydrate}</td></tr>
                      <tr><td style="font-weight:bold;">Fiber(/100g)</td><td>${fiber}</td></tr>
                    </table>
                  </div>
                </div>
                <div id="recognition-confirmed-tip" style="display:none;font-size:1.08em;color:#21a6b1;font-weight:bold;margin-bottom:10px;"></div>
                <div style="margin-top:32px; margin-bottom:12px;">
                  <form id="nutritionFormulaForm" style="display:flex; align-items:center; gap:12px; font-size:1.08em;">
                    <label style="font-weight:bold; min-width:48px;">(Calorie)</label>
                    <input type="number" id="calorieInput" style="width:70px; margin:0 6px; height:38px;" placeholder="" />
                    <span style="font-weight:bold; margin:0 4px;">×</span>
                    <label style="font-weight:bold; min-width:48px;">(Mass)</label>
                    <input type="number" id="massInput" style="width:70px; margin:0 6px; height:38px;" placeholder="" />
                    <span style="font-weight:bold; margin:0 4px;">=</span>
                    <input type="number" id="kcalInput" style="width:90px; margin:0 6px; height:38px;" placeholder="" />
                    <label style="font-weight:bold; min-width:48px;">(kcal)</label>
                    </form>
                </div>
                <div style="margin-bottom:8px; font-size:1.09em; font-weight:bold; color:#444;">
                  Is the automatic recognition result accurate?
                </div>
                <div style="display:flex; gap:18px; margin-bottom:18px;">
                  <button id="btnCorrect" style="background:#21a96b; color:#fff; padding:7px 22px; border-radius:7px; border:none; font-size:1.08em; font-weight:bold; cursor:pointer;">
                    Correct
                  </button>
                  <button id="btnIncorrect" style="background:#e74a3b; color:#fff; padding:7px 22px; border-radius:7px; border:none; font-size:1.08em; font-weight:bold; cursor:pointer;">
                    Incorrect
                  </button>
                </div>
                <div style="display:flex; flex-direction:column; gap:12px; align-items:flex-end; margin-bottom:20px;">
                  <button class="action-btn nutrition-back-btn"
                    id="btnBack"
                    style="background:#888; color:#fff; width:100px; border-radius:7px; border:none; font-size:1.07em; font-weight:bold; cursor:pointer;">
                    Back
                  </button>
                  <button class="action-btn nutrition-delete-btn"
                    id="btnDelete"
                    style="background:#4e73df; color:#fff; width:100px; border-radius:7px; border:none; font-size:1.07em; font-weight:bold; cursor:pointer;">
                    Delete
                  </button>
                </div>
                <div id="manualModal" style="display:none; position:fixed; left:0; top:0; width:100vw; height:100vh; background:rgba(0,0,0,0.24); z-index:9999; align-items:center; justify-content:center;">
                  <div style="background:#fff; border-radius:18px; max-width:400px; width:96vw; padding:32px 32px 24px 32px; box-shadow:0 10px 32px rgba(44,120,255,0.18); position:relative; display:flex; flex-direction:column; align-items:center;">
                    <div style="font-size:1.18em; font-weight:bold; margin-bottom:8px; align-self:flex-start;">Manual Recognition</div>
                    <div style="margin-bottom:10px; align-self:flex-start;">Food Name:</div>
                    <textarea id="manualFoodName" style="width:90%; min-height:80px; font-size:1.08em; border:1px solid #ccc; border-radius:7px; resize:vertical; padding:9px; margin-bottom:12px;"></textarea>
                    <div style="display:flex; justify-content:flex-end; gap:15px; margin-top:28px; width:90%;">
                      <button id="manualCancelBtn" style="background:#eee; color:#333; border-radius:7px; border:none; padding:7px 22px; font-size:1.08em; cursor:pointer;">
                        Cancel
                      </button>
                      <button id="manualOkBtn" style="background:#4e73df; color:#fff; border-radius:7px; border:none; padding:7px 22px; font-size:1.08em; cursor:pointer;">
                        Confirm
                      </button>
                    </div>
                  </div>
                </div>
            `;

            setTimeout(() => {
                const calorieInput = document.getElementById('calorieInput');
                const massInput = document.getElementById('massInput');
                const kcalInput = document.getElementById('kcalInput');

                // 自动填入
                if (massInput && typeof weight === 'number' && !isNaN(weight)) {
                    massInput.value = weight;
                } else if (massInput && weight && !isNaN(Number(weight))) {
                    massInput.value = Number(weight);
                }
                if (calorieInput && calorie && !isNaN(Number(calorie))) {
                    calorieInput.value = Number(calorie);
                }

                // 计算函数
                function updateKcal() {
                    const cal = parseFloat(calorieInput.value) || 0;
                    const mass = parseFloat(massInput.value) || 0;
                    if (!isNaN(cal) && !isNaN(mass)) {
                        kcalInput.value = cal * mass / 100 ;
                    } else {
                        kcalInput.value = '';
                    }
                }

                // 事件绑定
                calorieInput.addEventListener('input', updateKcal);
                massInput.addEventListener('input', updateKcal);
                kcalInput.addEventListener('input', function() {
                    const cal = parseFloat(calorieInput.value) || 0;
                    const kcalVal = parseFloat(kcalInput.value) || 0;
                    if (cal > 0) {
                        massInput.value = kcalVal / cal;
                    }
                });

                // 自动算一次
                updateKcal();

                // 手动识别弹窗
                document.getElementById('btnIncorrect').onclick = function() {
                    document.getElementById('manualModal').style.display = 'flex';
                };
                document.getElementById('manualCancelBtn').onclick = function() {
                    document.getElementById('manualModal').style.display = 'none';
                };
                document.getElementById('manualOkBtn').onclick = async function() {
                    const manualValue = document.getElementById('manualFoodName').value.trim();
                    if (manualValue.length === 0) {
                        alert('Please enter a food name!');
                        return;
                    }
                    try {
                        const resp = await fetch(`${API_BASE_URL}/updateFood`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                photoId: photoId,
                                newName: manualValue
                            })
                        });
                        if (!resp.ok) throw new Error('Server error');
                        const respData = await resp.json();
                        document.getElementById('recognition-confirmed-tip').style.display = 'block';
                        document.getElementById('recognition-confirmed-tip').textContent = 'Recognition confirmed: ' + manualValue;
                        document.getElementById('food-name-label').textContent = manualValue;
                        foodName = manualValue;
                        updateMealLinkName(photoId, manualValue, mealType);
                        await renderAllPhotoLinks();
                        document.getElementById('manualModal').style.display = 'none';
                    } catch (err) {
                        alert('Failed to update food name!');
                    }
                };
                document.getElementById('btnCorrect').onclick = function() {
                    alert('Thank you for confirming!');
                };

                // Back 按钮只返回，不做任何处理
                document.getElementById('btnBack').onclick = function() {
                    document.getElementById('photo-detail').classList.remove('active');
                    document.getElementById('Nutrition Management').classList.add('active');
                };

                // Delete 按钮：原先Back按钮的删除逻辑
                document.getElementById('btnDelete').onclick = async function() {
                    document.getElementById('photo-detail').classList.remove('active');
                    document.getElementById('Nutrition Management').classList.add('active');
                    await fetch(`${API_BASE_URL}/deleteFoodPhoto`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ photoId })
                    });
                    removeMealLinkAndTableRow(photoId, mealType);
                };
            }, 0);
   
        })
        .catch(() => {
            const detailDiv = document.getElementById('photo-detail');
            detailDiv.innerHTML = `
                <button class="action-btn nutrition-back-btn"
                  onclick="document.getElementById('photo-detail').classList.remove('active');
                           document.getElementById('Nutrition Management').classList.add('active');"
                  style="margin-bottom:18px;">Back</button>
                <div style="color:red;padding:20px;">Photo or nutrition info not found!</div>
            `;
        });
}


// 新增：根据 photoId 替换对应餐段链接文本
function updateMealLinkName(photoId, newName, mealType) {
    // 找到对应餐段
    const mealLinksId = {
        'Breakfast': 'breakfastLinks',
        'Lunch': 'lunchLinks',
        'Dinner': 'dinnerLinks',
        'Snack': 'snackLinks'
    }[mealType];
    if (!mealLinksId) return;
    const linksDiv = document.getElementById(mealLinksId);
    if (!linksDiv) return;
    [...linksDiv.children].forEach(link => {
        if (link.dataset && link.dataset.photoId == photoId) {
            link.textContent = newName;
        }
    });
}

function refreshNutritionDatePicker(date) {
    nutritionDate = date;
    document.getElementById('nutrition-date-picker').value = formatDateInput(date);
    fetchNutritionData(date);
}

// 在初始化时绑定 Check 按钮
function initNutritionManagement() {
    setupHistoryDatePicker(
        'nutrition-date-picker',
        'prevNutritionDate',
        'nextNutritionDate',
        (date) => {
            nutritionDate = date;
            refreshNutritionDatePicker(date);
        }
    );
    refreshNutritionDatePicker(new Date());
    renderAllPhotoLinks();
    setupNutritionCheckBtn(); // 新增，绑定 Check 按钮事件
}
initNutritionManagement();

// ================== 新增: 运动模式模块 (Workout Mode) ==================
const startWorkoutBtn = document.getElementById('start-workout-btn');
const stopWorkoutBtn = document.getElementById('stop-workout-btn');
const startSection = document.getElementById('start-workout-section');
const statusSection = document.getElementById('status-display-section');
const exerciseTypeSelect = document.getElementById('exercise-type-select');
const startWorkoutModal = document.getElementById('start-workout-modal');
const stopWorkoutModal = document.getElementById('stop-workout-modal');
const confirmStartBtn = document.getElementById('confirm-start-btn');
const cancelStartBtn = document.getElementById('cancel-start-btn');
const confirmStopBtn = document.getElementById('confirm-stop-btn');
const cancelStopBtn = document.getElementById('cancel-stop-btn');

// 运动记录表相关
const filterSelect = document.getElementById('filter-type-select');
const recordsTableBody = document.getElementById('records-table-body');
const prevPageBtn = document.getElementById('prev-page-btn');
const nextPageBtn = document.getElementById('next-page-btn');
const pageInfoSpan = document.getElementById('page-info');

let workoutSessionId = null;
let realtimeDataInterval = null;
let workoutTimer = null;
let secondsElapsed = 0;
let runningMap = null;
let runningPolyline = null;
let runningMarkers = [];
let runningRoutePoints = []; // 记录本次running轨迹点

let currentPage = 1;
let totalPages = 1;
const recordsPerPage = 10;

// ========== 运动模式UI切换 ==========
function updateWorkoutDisplay() {
    const type = exerciseTypeSelect.value;
    const countCard = document.getElementById('count-card');
    const speedCard = document.getElementById('speed-card');
    const countLabel = document.getElementById('count-label');
    const runningMapContainer = document.getElementById('running-map-container');
    const distanceCard = document.getElementById('distance-card');
    const paceCard = document.getElementById('pace-card');   // 新增
    document.getElementById('exercise-title').textContent = type.charAt(0).toUpperCase() + type.slice(1) + ' Workout';

    countCard.style.display = 'none';
    speedCard.style.display = 'none';
    runningMapContainer.style.display = 'none';
    distanceCard.style.display = 'none';
    if (paceCard) paceCard.style.display = 'none'; // 新增

    if (type === 'pushup') {
        countLabel.textContent = 'Pushups';
        countCard.style.display = 'block';
    } else if (type === 'squat') {
        countLabel.textContent = 'Squats';
        countCard.style.display = 'block';
    } else if (type === 'running') {
        speedCard.style.display = 'block';
        distanceCard.style.display = 'block';
        runningMapContainer.style.display = 'block';
        if (paceCard) paceCard.style.display = 'block'; // 新增
        initRunningMap();
    }
}
exerciseTypeSelect.addEventListener('change', updateWorkoutDisplay);
updateWorkoutDisplay();

// ========== 运动开始/结束弹窗 ==========
startWorkoutBtn.onclick = function() {
    startWorkoutModal.style.display = 'flex';
};
confirmStartBtn.onclick = function() {
    startWorkoutModal.style.display = 'none';
    startWorkout();
};
cancelStartBtn.onclick = function() {
    startWorkoutModal.style.display = 'none';
};

stopWorkoutBtn.onclick = function() {
    stopWorkoutModal.style.display = 'flex';
};
confirmStopBtn.onclick = function() {
    stopWorkoutModal.style.display = 'none';
    stopWorkout();
};
cancelStopBtn.onclick = function() {
    stopWorkoutModal.style.display = 'none';
};

// ========== 开始运动 ==========
async function startWorkout() {
    const exerciseType = exerciseTypeSelect.value;
    try {
        const resp = await fetch(`${API_BASE_URL}/api/workout/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exercise_type: exerciseType })
        });
        const data = await resp.json();
        if (exerciseType === 'running') {
            runningRoutePoints = [];
            initRunningMap();
            startRunningRouteRealtime();
        }
        window.runningSessionStartTime = new Date().toISOString();
        if (resp.ok) {
            workoutSessionId = data.session_id;
            startSection.style.display = 'none';
            statusSection.classList.add('active');
            secondsElapsed = 0;
            resetStatusDisplay();
            startTimer();
            startRealtimeDataPolling();
            // alert('Workout Started!');
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert('Failed to start workout.');
        console.error('Error starting workout:', error);
    }
}

// ========== 结束运动 ==========
async function stopWorkout() {
    clearInterval(realtimeDataInterval);
    clearInterval(workoutTimer);
    try {
        const resp = await fetch(`${API_BASE_URL}/api/workout/stop`, {
            method: 'POST'
        });
        if (exerciseTypeSelect.value === 'running') {
            stopRunningRouteRealtime();
        }
        window.runningSessionStartTime = null;
        const data = await resp.json();
        if (resp.ok) {
            workoutSessionId = null;
            statusSection.classList.remove('active');
            startSection.style.display = 'flex';
            fetchWorkoutRecords();
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert('Failed to stop workout.');
        console.error('Error stopping workout:', error);
    }
}


let runningRouteTimer = null;
// 实时轮询后端 GPS 点
function startRunningRouteRealtime() {
    stopRunningRouteRealtime();
    fetchRunningRoute();
    runningRouteTimer = setInterval(fetchRunningRoute, 2000); // 每2秒轮询
}
function stopRunningRouteRealtime() {
    if (runningRouteTimer) clearInterval(runningRouteTimer);
    runningRouteTimer = null;
}

async function fetchRunningRoute() {
    // 只在running session期间请求
    if (!workoutSessionId) return;
    // 获取本次running session开始时间后所有GPS点
    try {
        // 你需要在startWorkout时保存session的开始时间（见后端）。
        const startTime = window.runningSessionStartTime; // see后端
        if (!startTime) return;
        // 拉取所有大于startTime的点
        const url = `${API_BASE_URL}/getRunningRoute?session_id=${workoutSessionId}&start_time=${encodeURIComponent(startTime)}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data && Array.isArray(data.route)) {
            runningRoutePoints = data.route;
            renderRunningRoute(runningRoutePoints);
        }
    } catch (e) {}
}

// ========== 计时 ==========
function startTimer() {
    const timerDisplay = document.getElementById('timer-display');
    timerDisplay.textContent = '00:00:00';
    workoutTimer = setInterval(() => {
        secondsElapsed++;
        const hours = Math.floor(secondsElapsed / 3600);
        const minutes = Math.floor((secondsElapsed % 3600) / 60);
        const seconds = secondsElapsed % 60;
        timerDisplay.textContent =
            `${hours.toString().padStart(2, '0')}:` +
            `${minutes.toString().padStart(2, '0')}:` +
            `${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

// ========== 实时数据轮询 ==========
function startRealtimeDataPolling() {
    fetchRealtimeData();
    realtimeDataInterval = setInterval(fetchRealtimeData, 3000); // 3秒轮询
}

async function fetchRealtimeData() {
    try {
        const resp = await fetch(`${API_BASE_URL}/api/workout/realtime`);
        const data = await resp.json();
        if (resp.ok && data.active) {
            document.getElementById('heart-rate-display').textContent = data.heart_rate ?? '--';
            const caloriesVal = (typeof data.calories === 'number') ? data.calories.toFixed(2) : data.calories;
            document.getElementById('calories-display').textContent = caloriesVal ?? '--';
            const exerciseType = data.exercise_type;
            if (exerciseType === 'pushup') {
                document.getElementById('count-display').textContent = data.pushup_count ?? 0;
            } else if (exerciseType === 'squat') {
                document.getElementById('count-display').textContent = data.squat_count ?? 0;
            } else if (exerciseType === 'running') {
                // 跑步 distance 显示整数
                document.getElementById('distance-display').textContent = (parseInt(data.distance) || 0) + ' m';
                document.getElementById('speed-display').textContent = (data.speed ?? 0) + ' m/s';
                document.getElementById('pace-display').textContent = (data.pace !== undefined && data.pace !== null && data.pace !== '--' ? data.pace : '--') + ' steps/min';
            }

            // 时间格式化
            if (data.duration !== undefined) {
                const timerDisplay = document.getElementById('timer-display');
                const d = Number(data.duration);
                const hours = Math.floor(d / 3600);
                const minutes = Math.floor((d % 3600) / 60);
                const seconds = d % 60;
                timerDisplay.textContent =
                    `${hours.toString().padStart(2, '0')}:` +
                    `${minutes.toString().padStart(2, '0')}:` +
                    `${seconds.toString().padStart(2, '0')}`;
            }
        }
    } catch (error) {
        console.error('Failed to fetch realtime data:', error);
    }
}

// ========== 运动状态UI重置 ==========
function resetStatusDisplay() {
    document.getElementById('timer-display').textContent = '00:00:00';
    document.getElementById('heart-rate-display').textContent = '--';
    document.getElementById('calories-display').textContent = '--';
    document.getElementById('count-display').textContent = '0';
    document.getElementById('speed-display').textContent = '--';
    if (runningPolyline && runningMap) {
        runningMap.removeLayer(runningPolyline);
        runningPolyline = null;
    }
}

// ========== 跑步地图（Leaflet） ==========
function initRunningMap() {
    const mapDiv = document.getElementById('running-map');
    if (!mapDiv) return;
    // 避免尺寸未初始化时创建地图
    if (!mapDiv.offsetWidth || !mapDiv.offsetHeight) {
        setTimeout(initRunningMap, 300);
        return;
    }
    if (runningMap) {
        runningMap.eachLayer(layer => {
            if (layer instanceof L.Polyline || layer instanceof L.Marker) runningMap.removeLayer(layer);
        });
        if (runningPolyline) runningPolyline = null;
        runningMarkers = [];
        runningMap.invalidateSize();
        return;
    }
    // 和Steps页面一样，使用Carto底图
    runningMap = L.map('running-map', {
        center: [1.2966, 103.7764],
        zoom: 15,
        zoomControl: true,
        attributionControl: true
    });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> contributors',
    maxZoom: 19
    }).addTo(runningMap);
}

function renderRunningRoute(points) {
    if (!runningMap) return;
    // 清除旧图层
    if (runningPolyline) {
        runningMap.removeLayer(runningPolyline);
        runningPolyline = null;
    }
    // 清除旧marker
    if (runningMarkers) runningMarkers.forEach(m => runningMap.removeLayer(m));
    runningMarkers = [];

    if (!points || points.length === 0) return;
    // 轨迹
    runningPolyline = L.polyline(points.map(p => [p.lat, p.lng]), {
        color: '#22c55e',
        weight: 4,
        opacity: 0.95,
        lineCap: 'round',
        lineJoin: 'round'
    }).addTo(runningMap);
    // 起点
    const startIcon = L.icon({
        iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41]
    });
    const endIcon = L.icon({
        iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-red.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41]
    });
    runningMarkers = [];
    if (points.length >= 0)
        runningMarkers.push(L.marker([points[0].lat, points[0].lng], {icon: startIcon, title: 'Start'}).addTo(runningMap));
    if (points.length > 1)
        runningMarkers.push(L.marker([points[points.length - 1].lat, points[points.length - 1].lng], {icon: endIcon, title: 'End'}).addTo(runningMap));
    // 缩放视图
    const bounds = L.latLngBounds(points.map(pt => [pt.lat, pt.lng]));
    runningMap.fitBounds(bounds, {padding: [30,30]});
    //setTimeout(() => runningMap.invalidateSize(), 80);
}


function startRunningRouteRealtime() {
    if (runningRouteTimer) clearInterval(runningRouteTimer);
    fetchRunningRoute();
    runningRouteTimer = setInterval(fetchRunningRoute, 2000);
}
function stopRunningRouteRealtime() {
    if (runningRouteTimer) clearInterval(runningRouteTimer);
    runningRouteTimer = null;
}

async function fetchRunningRoute() {
    try {
        const resp = await fetch(`${API_BASE_URL}/getRunningSessionRoute`);
        const data = await resp.json();
        if (data && Array.isArray(data.route)) {
            renderRunningRoute(data.route);
        }
    } catch (e) {}
}

// ========== 运动记录分页 ==========
async function fetchWorkoutRecords() {
    const exerciseType = filterSelect.value;
    const url = `${API_BASE_URL}/api/workout/records?page=${currentPage}&per_page=${recordsPerPage}&exercise_type=${exerciseType}`;
    recordsTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center;">Loading...</td></tr>';
    prevPageBtn.disabled = true;
    nextPageBtn.disabled = true;

    try {
        const resp = await fetch(url);
        const data = await resp.json();
        if (resp.ok) {
            renderRecordsTable(data.records);
            totalPages = data.total_pages;
            pageInfoSpan.textContent = `Page ${currentPage} of ${totalPages}`;
            prevPageBtn.disabled = currentPage <= 1;
            nextPageBtn.disabled = currentPage >= totalPages;
        } else {
            recordsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: red;">Failed to load records.</td></tr>`;
        }
    } catch (error) {
        console.error('Error fetching workout records:', error);
        recordsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: red;">Failed to connect to server.</td></tr>`;
    }
}
function renderRecordsTable(records) {
    recordsTableBody.innerHTML = '';
    if (records.length === 0) {
        recordsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #888;">No records found.</td></tr>`;
        return;
    }
    records.forEach(record => {
        const row = document.createElement('tr');
        // const caloriesText = (typeof record.calories === 'number') ? record.calories.toFixed(2) : record.calories;

        let detailsCell = record.details;
        // ★★★★★ 加入按钮
        if (record.exercise_type === 'running') {
            detailsCell += ` <button class="action-btn workout-route-btn" style="margin-left:7px; padding:3px 12px; font-size:0.99em;" data-workout-id="${record.id}">View Route</button>`;
        }
        row.innerHTML = `
            <td>${record.date}</td>
            <td>${record.exercise_type.charAt(0).toUpperCase() + record.exercise_type.slice(1)}</td>
            <td>${record.duration}</td>
            <td>${detailsCell}</td>
            <td>${record.calories}</td>
        `;
        recordsTableBody.appendChild(row);
    });
    // ★★★★ 绑定点击事件（只给running的按钮）
    document.querySelectorAll('.workout-route-btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
            const workoutId = this.getAttribute('data-workout-id');
            showWorkoutRouteModal(workoutId);
        });
    });
}

// --------- 运动轨迹弹窗 ---------
let workoutRouteModal = null;
let workoutRouteMap = null;
let workoutRoutePolyline = null;
let workoutRouteMarkers = [];

function showWorkoutRouteModal(workoutId) {
    // 创建弹窗
    if (!workoutRouteModal) {
        workoutRouteModal = document.createElement('div');
        workoutRouteModal.className = 'modal';
        workoutRouteModal.id = 'workout-route-modal';
        workoutRouteModal.innerHTML = `
            <div class="modal-content" style="max-width:600px; width:96vw; min-height:330px; padding: 28px;">
                <h3 class="modal-title" style="margin-bottom:14px;">Running Route</h3>
                <div id="workout-route-map-container" style="width:100%; height:320px; margin-bottom:10px;">
                    <div id="workout-route-map" style="width:100%; height:100%;"></div>
                </div>
                <button class="action-btn" id="close-workout-route-modal" style="margin-top:8px;">Close</button>
            </div>
        `;
        document.body.appendChild(workoutRouteModal);
        // 关闭按钮
        document.getElementById('close-workout-route-modal').onclick = function () {
            workoutRouteModal.style.display = 'none';
            // 清除地图对象
            if (workoutRouteMap) {
                workoutRouteMap.remove();
                workoutRouteMap = null;
                workoutRoutePolyline = null;
                workoutRouteMarkers = [];
            }
        };
    }
    workoutRouteModal.style.display = 'flex';

    // 加载地图和数据
    setTimeout(() => {
        renderWorkoutRouteMap(workoutId);
    }, 150);
}

async function renderWorkoutRouteMap(workoutId) {
    // 初始化地图
    if (!workoutRouteMap) {
        workoutRouteMap = L.map('workout-route-map').setView([31.2304, 121.4737], 15);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://carto.com/">CARTO</a> contributors',
            maxZoom: 19
        }).addTo(workoutRouteMap);
    }
    // 清除旧图层
    if (workoutRoutePolyline) {
        workoutRouteMap.removeLayer(workoutRoutePolyline);
        workoutRoutePolyline = null;
    }
    if (workoutRouteMarkers) {
        workoutRouteMarkers.forEach(m => workoutRouteMap.removeLayer(m));
        workoutRouteMarkers = [];
    }
    // 拉数据
    try {
        const resp = await fetch(`${API_BASE_URL}/getWorkoutRouteById?id=${encodeURIComponent(workoutId)}`);
        const data = await resp.json();
        const points = data.route || [];
        if (!points.length) {
            // 没数据
            return;
        }
        // 绘制轨迹
        const latlngs = points.map(p => [p.lat, p.lng]);
        workoutRoutePolyline = L.polyline(latlngs, { color: '#22c55e', weight: 4 }).addTo(workoutRouteMap);
        // 起终点
        const startIcon = L.icon({ iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-green.png', iconSize: [25, 41], iconAnchor: [12, 41] });
        const endIcon = L.icon({ iconUrl: 'https://cdn.jsdelivr.net/gh/pointhi/leaflet-color-markers@master/img/marker-icon-red.png', iconSize: [25, 41], iconAnchor: [12, 41] });
        workoutRouteMarkers = [];
        if (latlngs.length > 0) workoutRouteMarkers.push(L.marker(latlngs[0], {icon: startIcon, title: 'Start'}).addTo(workoutRouteMap));
        if (latlngs.length > 1) workoutRouteMarkers.push(L.marker(latlngs[latlngs.length-1], {icon: endIcon, title: 'End'}).addTo(workoutRouteMap));
        workoutRouteMap.fitBounds(workoutRoutePolyline.getBounds(), {padding: [18,18]});
        setTimeout(() => workoutRouteMap.invalidateSize(), 120);
    } catch (err) {}
}


// 记录筛选与分页
filterSelect.addEventListener('change', () => {
    currentPage = 1;
    fetchWorkoutRecords();
});
prevPageBtn.addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        fetchWorkoutRecords();
    }
});
nextPageBtn.addEventListener('click', () => {
    if (currentPage < totalPages) {
        currentPage++;
        fetchWorkoutRecords();
    }
});

// 页面初始加载
fetchWorkoutRecords();
});
    const measureBtn = document.getElementById('nutritionNewMeasureBtn');
    // Nutrition Mass Popup: Four-Step Flow
    document.getElementById('nutritionNewMeasureBtn').addEventListener('click', function(e) {

        if (measureBtn.disabled) {
            // 可以弹窗、或者弹提示
            alert('Please upload a food photo first!');
            e.preventDefault();
            return false;
        }
        const nutritionMassPopup = document.getElementById('nutritionMassPopup');
        let emptyTrayWeight = null;
        let foodTrayWeight = null;

        // Step 1: Remove items
        nutritionMassPopup.innerHTML = `
          <div class="popup-content">
            <div class="step-label">Step 1: Remove items</div>
            <div style="margin:12px 0;">Please make sure the scale is empty. Remove all items and click Confirm.</div>
            <div class="popup-footer">
              <button class="popup-btn confirm" id="btnStep1Confirm">Confirm</button>
            </div>
          </div>
        `;
        nutritionMassPopup.style.display = 'flex';
        document.body.style.overflow = "hidden";

        // Step 1 → Step 2
        nutritionMassPopup.querySelector('#btnStep1Confirm').onclick = function() {
        nutritionMassPopup.innerHTML = `
          <div class="popup-content">
            <div class="step-label">Step 2: Switch to food mode</div>
            <div style="margin:12px 0;">Click the button below to switch (the scale will reset and record a new entry).</div>
            <div class="popup-footer">
              <button class="popup-btn" id="btnSwitchFoodMode">Switch to food mode</button>
            </div>
            <div class="popup-msg" id="step2Msg"></div>
          </div>
        `;
        const btn = nutritionMassPopup.querySelector('#btnSwitchFoodMode');
        const step2Msg = nutritionMassPopup.querySelector('#step2Msg');
        step2Msg.textContent = 'Checking scale mode...';

        // 每次点击都重新请求后端
        btn.onclick = async function() {
            step2Msg.textContent = 'Checking scale mode...';
            try {
                const res = await fetch(`${API_BASE_URL}/getScaleMode`);
                const data = await res.json();
                if (data.last_mode === 'food') {
                    step2Msg.textContent = 'Scale is in food mode. You can continue.';
                    setTimeout(() => showStep3(), 700);
                } else if (data.last_mode === 'water') {
                    step2Msg.textContent = 'Scale is still in water mode. Please finish water weighing, then try again.';
                    // 按钮一直可点，用户可以多次点击重试
                } else {
                    step2Msg.textContent = 'Unable to determine scale mode. Please check connection or data.';
                }
            } catch (err) {
                step2Msg.textContent = 'Failed to check scale mode. Please retry.';
            }
        };
    };

    // Step 3: Weigh empty tray
    function showStep3() {
        nutritionMassPopup.innerHTML = `
          <div class="popup-content">
            <div class="step-label">Step 3: Weigh empty tray</div>
            <div style="margin:12px 0;">Place the empty tray on the scale and click "Measure" to display the current weight.</div>
            <div>
              <button class="popup-btn" id="btnMeasureEmptyTray">Measure</button>
              <span class="popup-value" id="popupEmptyTrayValue">--</span>
            </div>
            <div class="popup-footer" style="margin-top:20px;">
              <button class="popup-btn confirm" id="btnStep3Next">Next Step</button>
            </div>
            <div class="popup-msg" id="step3Msg"></div>
          </div>
        `;
        nutritionMassPopup.querySelector('#btnMeasureEmptyTray').onclick = async function() {
            const valSpan = nutritionMassPopup.querySelector('#popupEmptyTrayValue');
            const msgSpan = nutritionMassPopup.querySelector('#step3Msg');
            msgSpan.textContent = 'Measuring...';
            try {
                const res = await fetch(`${API_BASE_URL}/getLatestScaleWeight?mode=food`);
                const data = await res.json();
                if (typeof data.weight === 'number') {
                    emptyTrayWeight = Number(data.weight);
                    valSpan.textContent = `${emptyTrayWeight} g`;
                    msgSpan.textContent = 'Measured successfully!';
                } else {
                    msgSpan.textContent = 'Failed to measure, try again.';
                }
            } catch {
                msgSpan.textContent = 'Connection failed, please try again.';
            }
        };
        nutritionMassPopup.querySelector('#btnStep3Next').onclick = function() {
            showStep4();
        };
    }

    // Step 4: Place food and measure
    function showStep4() {
        nutritionMassPopup.innerHTML = `
          <div class="popup-content">
            <div class="step-label">Step 4: Place food and measure</div>
            <div style="margin:12px 0;">Place the food on the tray, then click "Measure" to display the current weight.</div>
            <div>
              <button class="popup-btn" id="btnMeasureWithFood">Measure</button>
              <span class="popup-value" id="popupWithFoodValue">--</span>
            </div>
            <div class="popup-footer" style="margin-top:20px;">
              <button class="popup-btn confirm" id="btnStep4Confirm">Confirm</button>
              <button class="popup-btn cancel" id="btnStep4Cancel">Cancel</button>
            </div>
            <div class="popup-msg" id="step4Msg"></div>
          </div>
        `;
        nutritionMassPopup.querySelector('#btnMeasureWithFood').onclick = async function() {
            const valSpan = nutritionMassPopup.querySelector('#popupWithFoodValue');
            const msgSpan = nutritionMassPopup.querySelector('#step4Msg');
            msgSpan.textContent = 'Measuring...';
            try {
                const res = await fetch(`${API_BASE_URL}/getLatestScaleWeight?mode=food`);
                const data = await res.json();
                if (typeof data.weight === 'number') {
                    foodTrayWeight = Number(data.weight);
                    valSpan.textContent = `${foodTrayWeight} g`;
                    msgSpan.textContent = 'Measured successfully!';
                } else {
                    msgSpan.textContent = 'Failed to measure, try again.';
                }
            } catch {
                msgSpan.textContent = 'Connection failed, please try again.';
            }
        };

        nutritionMassPopup.querySelector('#btnStep4Confirm').onclick = async function() {
            const msgSpan = nutritionMassPopup.querySelector('#step4Msg');
            if (emptyTrayWeight === null || foodTrayWeight === null) {
                msgSpan.textContent = 'Please complete both measurements first!';
                return;
            }
            const massVal = foodTrayWeight - emptyTrayWeight;

            const massInputElem = document.getElementById('massInput');
            if (massInputElem) {
                massInputElem.value = massVal > 0 ? massVal : 0;
            }
            if (typeof updateKcal === 'function') updateKcal();

            msgSpan.textContent = 'Saving weight...';

            try {
                const resp = await fetch(`${API_BASE_URL}/setLatestFoodPhotoWeight`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ weight: massVal > 0 ? massVal : 0 })
                });
                const data = await resp.json();
                if (resp.ok && data && data.weight !== undefined) {
                    msgSpan.textContent = 'Weight saved!';
                    setTimeout(() => {
                        nutritionMassPopup.style.display = 'none';
                        document.body.style.overflow = "";
                    }, 600);
                } else {
                    msgSpan.textContent = 'Failed to save weight (no photo row)!';
                }
            } catch (err) {
                msgSpan.textContent = 'Failed to save weight! Please check network or upload photo first.';
            }
        };

        nutritionMassPopup.querySelector('#btnStep4Cancel').onclick = function() {
            nutritionMassPopup.style.display = 'none';
            document.body.style.overflow = "";
        };
    }
});
// 喝水历史选择器和Check按钮逻辑
function formatDateInput(d) {
    return d.toISOString().substring(0,10);
}
function addDays(dateStr, offset) {
    const d = new Date(dateStr);
    d.setDate(d.getDate() + offset);
    return formatDateInput(d);
}
function refreshWater(date) {
    const dateStr = formatDateInput(date || new Date());
    fetch(`${API_BASE_URL}/getWaterAmount?date=${encodeURIComponent(dateStr)}`)
        .then(res => res.json())
        .then(data => {
            document.getElementById('water-today').textContent = data.amount ? (data.amount + 'ml') : '--';
        })
        .catch(() => {
            document.getElementById('water-today').textContent = '--';
        });
}

async function refreshHydrationSuggestion(selectedDate) {
    const suggestionDiv = document.getElementById('water-suggestion');
    const todayStr = formatDateInput(new Date());
    const selectedDateStr = formatDateInput(selectedDate);

    if (selectedDateStr === todayStr) {
        // 当天：显示目标减去已喝量
        try {
            const [amountRes, targetRes] = await Promise.all([
                fetch(`${API_BASE_URL}/getWaterAmount?date=${encodeURIComponent(todayStr)}`),
                fetch(`${API_BASE_URL}/getWaterTarget`)
            ]);
            const amountData = await amountRes.json();
            const targetData = await targetRes.json();
            const todayAmount = Number(amountData.amount) || 0;
            const target = Number(targetData.target) || 2000;
            if (todayAmount >= target) {
                suggestionDiv.textContent = `Goal achieved 🎉！`;
                suggestionDiv.style.color = '#21a96b';
            } else {
                const left = target - todayAmount;
                suggestionDiv.textContent = `You still need to drink ${left} ml of water to reach your goal.`;
                suggestionDiv.style.color = '#21a96b';
            }
        } catch (err) {
            suggestionDiv.textContent = `Drink more water please ~`;
            suggestionDiv.style.color = '#888';
        }
    } else {
        // 非当天：显示固定建议
        suggestionDiv.textContent = `Drinking more water is good for your health.`;
        suggestionDiv.style.color = '#21a96b';
    }
}

// 在 hydration-section 显示时调用该函数
document.getElementById('hydration-section').addEventListener('transitionend', function() {
    if (this.classList.contains('active')) {
        refreshHydrationSuggestion();
    }
});
// 或在每次日期变更、Check按钮后都调用

function updateHydrationSectionLayout(selectedDate) {
    const todayStr = formatDateInput(new Date());
    const selectedDateStr = formatDateInput(selectedDate);

    const setTargetCard = document.querySelector('#hydration-section .center-form-card');
    if (selectedDateStr === todayStr) {
        setTargetCard?.classList.remove('hidden');
        // 新增：自动填入目标
        fetch(`${API_BASE_URL}/getWaterTarget`)
            .then(res => res.json())
            .then(data => {
                const input = document.getElementById('waterTargetInput');
                if (input && data.target) input.value = data.target;
            });
    } else {
        setTargetCard?.classList.add('hidden');
    }
    refreshHydrationSuggestion(selectedDate);
}

function setupWaterHistoryPicker() {
    const input = document.getElementById('water-history-date');
    const prevBtn = document.getElementById('prevWaterDate');
    const nextBtn = document.getElementById('nextWaterDate');

    let waterRealtimeTimer = null;

    if (!input) return;
    input.value = formatDateInput(new Date());

    function onDateChangeHandler() {
        const selectedDate = new Date(input.value);
        updateHydrationSectionLayout(selectedDate);
        refreshWater(selectedDate);
        refreshHydrationSuggestion(selectedDate); // 保证切换时也刷新建议

        if (waterRealtimeTimer) {
            clearInterval(waterRealtimeTimer);
            waterRealtimeTimer = null;
        }
        if (formatDateInput(selectedDate) === formatDateInput(new Date())) {
            waterRealtimeTimer = setInterval(() => {
                refreshWater(selectedDate);
                refreshHydrationSuggestion(selectedDate); // 新增：自动刷新建议
            }, 3000); // 每3秒刷新一次
        }
    }

    prevBtn?.addEventListener('click', () => {
        input.value = addDays(input.value, -1);
        onDateChangeHandler();
    });
    nextBtn?.addEventListener('click', () => {
        input.value = addDays(input.value, 1);
        onDateChangeHandler();
    });
    input.addEventListener('change', onDateChangeHandler);

    onDateChangeHandler();

    fetch(`${API_BASE_URL}/getWaterTarget`)
        .then(res => res.json())
        .then(data => {
            const input = document.getElementById('waterTargetInput');
            if (input && data.target) input.value = data.target;
        });
}



async function refreshSedentaryCircle() {
    try {
        const res = await fetch(`${API_BASE_URL}/getSedentaryStatus`);
        const data = await res.json();
        const standHours = data.stand_hours || [];
        const totalStand = data.total_stand || 0;

        // 渲染24个点，只清空点的容器
        const dotsDiv = document.getElementById('sedentary-dots');
        dotsDiv.innerHTML = '';
        const r = 38; // 半径
        const cx = 46, cy = 46; // 圆心
        for (let i = 0; i < 24; i++) {
            const angle = (i / 24) * 2 * Math.PI - Math.PI / 2;
            const x = cx + r * Math.cos(angle) - 5.5;
            const y = cy + r * Math.sin(angle) - 5.5;
            const dot = document.createElement('div');
            dot.className = 'sedentary-dot ' + (standHours.includes(i) ? 'active' : 'inactive');
            dot.style.left = `${x}px`;
            dot.style.top = `${y}px`;
            dot.title = `${i}:00`;
            dot.style.position = "absolute";
            dotsDiv.appendChild(dot);
        }

        // 圆心数字
        document.getElementById('sedentary-center-num').textContent = totalStand;

        // 占比计算
        const now = new Date();
        // const hourNow = now.getHours() + 1; // 已经过的小时数（含当前小时，防0/0）
        const ratio = totalStand / 12;
        const cardDiv = document.getElementById('home-sedentary');
        cardDiv.classList.remove('sedentary-red', 'sedentary-yellow', 'sedentary-green');
        let color = '';
        if (ratio < 0.5) {
            color = 'sedentary-red';
        } else if (ratio < 0.75) {
            color = 'sedentary-yellow';
        } else {
            color = 'sedentary-green';
        }
        cardDiv.classList.add(color);

        // 可选：中间数字颜色也跟随状态
        document.getElementById('sedentary-center-num').style.color =
            (color === 'sedentary-red') ? '#e74a3b' :
            (color === 'sedentary-yellow') ? '#f6c23e' : '#1cc88a';

        // 提示语也可自定义
        document.getElementById('home-sedentary-tip').textContent =
            (color === 'sedentary-red') ? 'Move more! Stand up to stay healthy.' :
            (color === 'sedentary-yellow') ? 'Good job! Try to stand more.' :
            'Excellent! You\'re active today!';
    } catch (err) {
        // 错误处理
        document.getElementById('sedentary-center-num').textContent = '--';
    }
}

// 页面初始和每小时定时刷新
refreshSedentaryCircle();
setInterval(refreshSedentaryCircle, 60 * 60 * 1000);
