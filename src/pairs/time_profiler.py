import time
from typing import Dict, Optional


class StageProfiler:
    """
    Простая утилита для профилирования времени выполнения этапов.
    
    Использование:
        profiler = StageProfiler(show=True)
        profiler.start_stage("Этап 1", "Вычисление таблиц")
        # ... код этапа ...
        profiler.end_stage("Этап 1")
        
        summary = profiler.get_summary()
    """
    
    def __init__(self, show: bool = True):
        """
        Args:
            show: если True, выводит информацию о каждом этапе
        """
        self.show = show
        self.stages: Dict[str, dict] = {}
        self.current_stage: Optional[str] = None
        self.start_time: Optional[float] = None
        self.total_start_time: Optional[float] = None
        
    def start_profiling(self):
        """Начинает общее профилирование"""
        self.total_start_time = time.time()
        if self.show:
            print("🚀 ПРОФИЛИРОВАНИЕ НАЧАТО")
            print("=" * 50)
    
    def start_stage(self, stage_name: str, description: str = ""):
        """
        Начинает замер времени для этапа
        
        Args:
            stage_name: уникальное имя этапа
            description: описание этапа для вывода
        """
        if self.current_stage is not None:
            if self.show:
                print(f"⚠️  Предыдущий этап '{self.current_stage}' не был завершен!")
        
        self.current_stage = stage_name
        self.start_time = time.time()
        
        # Инициализируем запись этапа
        self.stages[stage_name] = {
            'description': description,
            'start_time': self.start_time,
            'end_time': None,
            'duration': None,
            'status': 'running'
        }
        
        if self.show:
            stage_info = f" - {description}" if description else ""
            print(f"⏱️  {stage_name}{stage_info}...", end=" ", flush=True)
    
    def end_stage(self, stage_name: str, details: str = ""):
        """
        Завершает замер времени для этапа
        
        Args:
            stage_name: имя этапа (должно совпадать с start_stage)
            details: дополнительная информация о результате
        """
        if self.current_stage != stage_name:
            if self.show:
                print(f"⚠️  Попытка завершить этап '{stage_name}', но текущий этап '{self.current_stage}'")
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Обновляем запись этапа
        self.stages[stage_name].update({
            'end_time': end_time,
            'duration': duration,
            'status': 'completed',
            'details': details
        })
        
        self.current_stage = None
        
        if self.show:
            details_info = f" ({details})" if details else ""
            print(f"✅ {self._format_duration(duration)}{details_info}")
    
    def fail_stage(self, stage_name: str, error_msg: str = ""):
        """
        Отмечает этап как неудачный
        
        Args:
            stage_name: имя этапа
            error_msg: сообщение об ошибке
        """
        if self.current_stage == stage_name:
            end_time = time.time()
            duration = end_time - self.start_time
            
            self.stages[stage_name].update({
                'end_time': end_time,
                'duration': duration,
                'status': 'failed',
                'error': error_msg
            })
            
            self.current_stage = None
            
            if self.show:
                error_info = f" - {error_msg}" if error_msg else ""
                print(f"❌ {self._format_duration(duration)}{error_info}")
    
    def get_summary(self) -> Dict:
        """
        Возвращает сводку по всем этапам
        
        Returns:
            dict: словарь с информацией о временах и статистике
        """
        total_duration = 0
        completed_stages = 0
        failed_stages = 0
        
        stage_times = {}
        
        for stage_name, stage_info in self.stages.items():
            if stage_info['duration'] is not None:
                total_duration += stage_info['duration']
                stage_times[stage_name] = stage_info['duration']
                
                if stage_info['status'] == 'completed':
                    completed_stages += 1
                elif stage_info['status'] == 'failed':
                    failed_stages += 1
        
        # Общее время с момента начала профилирования
        total_elapsed = None
        if self.total_start_time is not None:
            total_elapsed = time.time() - self.total_start_time
        
        return {
            'total_stages': len(self.stages),
            'completed_stages': completed_stages,
            'failed_stages': failed_stages,
            'total_duration': total_duration,
            'total_elapsed': total_elapsed,
            'stage_times': stage_times,
            'stages_detail': self.stages.copy()
        }
    
    def print_summary(self):
        """Выводит детальную сводку профилирования"""
        summary = self.get_summary()
        
        if not self.show:
            return
        
        print("\n" + "=" * 50)
        print("📊 СВОДКА ПРОФИЛИРОВАНИЯ")
        print("=" * 50)
        
        # Общая статистика
        print(f"Всего этапов: {summary['total_stages']}")
        print(f"Завершено успешно: {summary['completed_stages']}")
        print(f"Завершено с ошибкой: {summary['failed_stages']}")
        
        if summary['total_elapsed']:
            print(f"Общее время: {self._format_duration(summary['total_elapsed'])}")
        if summary['total_duration'] > 0:
            print(f"Время этапов: {self._format_duration(summary['total_duration'])}")
        
        print()
        
        # Детали по этапам
        if summary['stage_times']:
            print("📈 Время по этапам:")
            for stage_name, duration in summary['stage_times'].items():
                stage_info = self.stages[stage_name]
                percentage = (duration / summary['total_duration']) * 100 if summary['total_duration'] > 0 else 0
                
                status_icon = "✅" if stage_info['status'] == 'completed' else "❌"
                details = stage_info.get('details', '')
                details_str = f" - {details}" if details else ""
                
                print(f"  {status_icon} {stage_name}: {self._format_duration(duration)} ({percentage:.1f}%){details_str}")
        
        print("=" * 50)
    
    def _format_duration(self, duration: float) -> str:
        """Форматирует время в удобочитаемый вид"""
        if duration < 0.001:
            return f"{duration * 1000000:.0f}μs"
        elif duration < 1.0:
            return f"{duration * 1000:.1f}ms"
        elif duration < 60:
            return f"{duration:.2f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m {seconds:.1f}s"