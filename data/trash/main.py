"""
KRONOS Sequential Processing Pipeline
======================================

Complete pipeline:
1. Load data (data_loader.py)
2. Load model (app.py)
3. Sequential batch processing (sequential.py → app.py)
4. Results aggregation
"""

from src.data.data_loader import load_csv_data
from src.app import load_model, predict_from_dataframe
from src.sequential import SequentialProcessor, WindowConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from datetime import datetime
import json
import time


console = Console()


def print_header():
    """Print beautiful header"""
    header = Text()
    header.append("\n")
    header.append("╔═══════════════════════════════════════════════════════════════════╗\n", style="bold cyan")
    header.append("║                                                                   ║\n", style="bold cyan")
    header.append("║           ", style="bold cyan")
    header.append("🚀 KRONOS SEQUENTIAL PROCESSING PIPELINE 🚀", style="bold yellow")
    header.append("          ║\n", style="bold cyan")
    header.append("║                                                                   ║\n", style="bold cyan")
    header.append("╚═══════════════════════════════════════════════════════════════════╝\n", style="bold cyan")
    console.print(header)


def create_info_table(title, data, icon="📋"):
    """Create a formatted info table"""
    table = Table(title=f"{icon} {title}", box=box.ROUNDED, show_header=False, 
                  border_style="cyan", title_style="bold cyan")
    table.add_column("Property", style="dim", width=20)
    table.add_column("Value", style="bold green")
    
    for key, value in data.items():
        table.add_row(key, str(value))
    
    return table


def create_stats_panel(stats):
    """Create live statistics panel"""
    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    
    stats_table = Table(box=box.HEAVY, show_header=True, header_style="bold magenta",
                       border_style="magenta", expand=True)
    stats_table.add_column("Metric", style="cyan", justify="left")
    stats_table.add_column("Value", style="bold green", justify="right")
    
    stats_table.add_row("✅ Successful", str(stats['successful']))
    stats_table.add_row("❌ Failed", str(stats['failed']))
    stats_table.add_row("📈 Predictions", str(stats['predictions']))
    stats_table.add_row("⚡ Avg Time/Batch", f"{stats['avg_time']:.2f}s")
    
    grid.add_row(stats_table)
    
    return Panel(
        grid,
        title="[bold yellow]⚡ Live Statistics ⚡[/bold yellow]",
        border_style="yellow",
        padding=(1, 2)
    )


def main():
    start_time = time.time()
    
    print_header()
    
    # Load configuration
    with console.status("[bold cyan]Loading configuration...", spinner="dots"):
        time.sleep(0.5)  # Visual feedback
        with open('config.json', 'r') as f:
            config = json.load(f)
    
    console.print("✓ Configuration loaded\n", style="bold green")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    console.print(Panel.fit(
        "[bold white]STEP 1: Loading Data[/bold white]",
        border_style="blue",
        padding=(0, 2)
    ))
    
    with console.status("[bold blue]Reading CSV data...", spinner="bouncingBall"):
        df = load_csv_data(file_path=config['data']['file_path'])
    
    data_info = {
        "Rows": f"{len(df):,}",
        "Columns": f"{len(df.columns)}",
        "Features": ", ".join(df.columns.tolist()),
        "Date Range": f"{df.index[0]} → {df.index[-1]}",
        "File": config['data']['file_path']
    }
    
    console.print(create_info_table("Data Loaded Successfully", data_info, "📊"))
    console.print()
    
    # ========================================================================
    # STEP 2: Load Model
    # ========================================================================
    console.print(Panel.fit(
        "[bold white]STEP 2: Loading Model[/bold white]",
        border_style="magenta",
        padding=(0, 2)
    ))
    
    with console.status(f"[bold magenta]Loading {config['model']['model_key']} on {config['model']['device']}...", 
                       spinner="dots12"):
        load_model(
            model_key=config['model']['model_key'], 
            device=config['model']['device']
        )
    
    model_info = {
        "Model": config['model']['model_key'],
        "Device": config['model']['device'].upper(),
        "Status": "Ready ✓"
    }
    
    console.print(create_info_table("Model Configuration", model_info, "🤖"))
    console.print()
    
    # ========================================================================
    # STEP 3: Configure Sequential Processing
    # ========================================================================
    console.print(Panel.fit(
        "[bold white]STEP 3: Configuring Processor[/bold white]",
        border_style="green",
        padding=(0, 2)
    ))
    
    window_config = WindowConfig(
        window_size=config['sequential_processing']['window_size'],
        step_size=config['sequential_processing']['step_size']
    )
    
    processor = SequentialProcessor(window_config)
    total_batches = processor.calculate_total_batches(len(df))
    
    proc_info = {
        "Window Size": f"{window_config.window_size:,} rows",
        "Step Size": f"{window_config.step_size:,} rows",
        "Total Batches": str(total_batches),
        "Lookback": str(config['prediction']['lookback']),
        "Pred Length": str(config['prediction']['pred_len']),
        "Samples/Pred": str(config['prediction']['sample_count'])
    }
    
    console.print(create_info_table("Processing Configuration", proc_info, "⚙️"))
    console.print()
    
    # ========================================================================
    # STEP 4: Process Batches Sequentially
    # ========================================================================
    console.print(Panel.fit(
        "[bold white]STEP 4: Processing Batches[/bold white]",
        border_style="yellow",
        padding=(0, 2)
    ))
    
    # Statistics tracking
    stats = {
        'successful': 0,
        'failed': 0,
        'predictions': 0,
        'avg_time': 0.0,
        'batch_times': []
    }
    
    # Create progress bar
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        
        task = progress.add_task("[cyan]Processing batches...", total=total_batches)
        
        for result in processor.process_batches(
            df=df,
            predict_func=predict_from_dataframe,
            lookback=config['prediction']['lookback'],
            pred_len=config['prediction']['pred_len'],
            temperature=config['prediction']['temperature'],
            top_p=config['prediction']['top_p'],
            sample_count=config['prediction']['sample_count'],
            group=config['prediction']['group']
        ):
            batch_start = time.time()
            batch_num = result['batch_info']['batch_number']
            
            if result['status'] == 'success':
                stats['successful'] += 1
                pred_count = len(result['predictions'])
                stats['predictions'] += pred_count
                
                # Update description with success
                progress.update(
                    task, 
                    description=f"[green]✓ Batch {batch_num}/{total_batches}[/green] [dim]({pred_count} preds)[/dim]",
                    advance=1
                )
            else:
                stats['failed'] += 1
                error_msg = result['error'][:30] + "..." if len(result['error']) > 30 else result['error']
                
                # Update description with error
                progress.update(
                    task,
                    description=f"[red]✗ Batch {batch_num}/{total_batches}[/red] [dim]({error_msg})[/dim]",
                    advance=1
                )
            
            batch_time = time.time() - batch_start
            stats['batch_times'].append(batch_time)
            stats['avg_time'] = sum(stats['batch_times']) / len(stats['batch_times'])
    
    console.print()
    
    # Show live stats after processing
    console.print(create_stats_panel(stats))
    console.print()
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    console.print(Panel.fit(
        "[bold white]STEP 5: Generating Summary[/bold white]",
        border_style="cyan",
        padding=(0, 2)
    ))
    
    all_results = processor.get_all_results()
    successful = processor.get_successful_results()
    failed = processor.get_failed_results()
    
    # Create summary table
    summary_table = Table(title="📊 Processing Summary", box=box.DOUBLE_EDGE, 
                         show_header=True, header_style="bold yellow",
                         border_style="yellow")
    
    summary_table.add_column("Metric", style="cyan", justify="left", width=30)
    summary_table.add_column("Value", style="bold white", justify="right", width=20)
    summary_table.add_column("Status", justify="center", width=15)
    
    total_preds = sum(len(r['predictions']) for r in successful) if successful else 0
    success_rate = (len(successful) / len(all_results) * 100) if all_results else 0
    
    summary_table.add_row("Total Batches", str(len(all_results)), "📦")
    summary_table.add_row("Successful Batches", str(len(successful)), "✅" if success_rate == 100 else "⚠️")
    summary_table.add_row("Failed Batches", str(len(failed)), "❌" if failed else "✅")
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%", 
                         "✅" if success_rate == 100 else "⚠️")
    summary_table.add_row("Total Predictions", f"{total_preds:,}", "📈")
    summary_table.add_row("Avg Time/Batch", f"{stats['avg_time']:.2f}s", "⚡")
    summary_table.add_row("Total Time", f"{time.time() - start_time:.2f}s", "⏱️")
    
    console.print(summary_table)
    console.print()
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_batches': len(all_results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': success_rate,
        'total_predictions': total_preds,
        'avg_batch_time': stats['avg_time'],
        'total_time': time.time() - start_time,
        'config': {
            'window_size': window_config.window_size,
            'step_size': window_config.step_size,
            'model': config['model']['model_key'],
            'device': config['model']['device']
        },
        'results': [
            {
                'batch': r['batch_info']['batch_number'],
                'status': r['status'],
                'predictions_count': len(r['predictions']),
            }
            for r in all_results
        ]
    }
    
    output_file = config['output']['summary_file']
    with console.status(f"[bold cyan]Saving summary to {output_file}...", spinner="dots"):
        time.sleep(0.3)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    console.print(f"💾 [bold green]Summary saved to:[/bold green] [cyan]{output_file}[/cyan]\n")
    
    # Final message
    if success_rate == 100:
        final_msg = Panel(
            "[bold green]🎉 ALL BATCHES PROCESSED SUCCESSFULLY! 🎉[/bold green]",
            border_style="bold green",
            padding=(1, 2)
        )
    else:
        final_msg = Panel(
            f"[bold yellow]⚠️  PROCESSING COMPLETE WITH {len(failed)} FAILED BATCH(ES) ⚠️[/bold yellow]",
            border_style="bold yellow",
            padding=(1, 2)
        )
    
    console.print(final_msg)
    console.print()


if __name__ == '__main__':
    main()