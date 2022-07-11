
class Duration: 
    
    def __init__(self, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
    
    def in_seconds(self) -> int: 
        days = self.days
        hours = days * 24 + self.hours
        minutes = hours * 60 + self.minutes
        seconds = minutes * 60 + self.seconds
        return seconds