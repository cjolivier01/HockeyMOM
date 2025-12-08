import time
from rich.progress import Progress

def simulate_rich_task(iterations):
    with Progress() as progress:
        task1 = progress.add_task("[green]Downloading...", total=iterations)
        task2 = progress.add_task("[cyan]Processing...", total=iterations // 2)

        for i in range(iterations):
            time.sleep(0.05)
            progress.update(task1, advance=1)
            if i % 2 == 0:
                progress.update(task2, advance=1)

if __name__ == "__main__":
    simulate_rich_task(60)
    print("Rich tasks completed!")
