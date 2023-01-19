class Population:
    def __init__(self, **kwargs):
        self.N = kwargs.get('pop_size', 12000)
        
        self.VLdistribution = None
        
        self.infection_status = None # List[List]
        
    
    def generate_infection(self):
        pass
        
        
    def perform_group_testing(self):
        pass
    
    
    def compute_metrics(self):
        pass
        
        
        
        
        
        
        