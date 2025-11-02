import pandas as pd
import numpy as np
from aipe.interface.orchestrator import Orchestrator

if __name__ == "__main__":
    # Create a dummy dataset for demonstration purposes
    data = pd.DataFrame(np.random.rand(100, 3), columns=['X', 'Y', 'Z'])

    orchestrator = Orchestrator()
    orchestrator.run(data)
