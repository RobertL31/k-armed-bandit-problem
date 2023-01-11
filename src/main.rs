
use rand_distr::{Normal, Distribution};
use rand::Rng;
use std::{cmp::Ordering, collections::HashMap, hash::Hash};

use plotters::prelude::*;

const N_SAMPLE: u32 = 2000;
const N_STEPS: u32 = 1000;
const DISTRIBUTION_RANGE: f32 = 1.0;

struct Machine {
    distribution : Normal<f32>
}

impl Machine {
    fn new() -> Self {
        let shift : f32 = rand::thread_rng().gen_range(-DISTRIBUTION_RANGE, DISTRIBUTION_RANGE);
        Machine {
            distribution : Normal::new(0.0 + shift,1.0).unwrap()
        }
    }

    fn get_value(&self) -> f32 {
        self.distribution.sample(&mut rand::thread_rng())
    }
}

struct Model<'a> {
    machines: &'a Vec<Machine>,
    estimates : Vec<(f32,u32)>,
    exploration_rate : f32,
    scores : Vec<f32>,
    name : String
}

fn compare(x: f32, y: f32) -> Ordering {
    match x < y {
        true => Ordering::Less,
        false => Ordering::Greater
    }
}

impl<'a> Model<'a> {
    fn new(machines: &'a Vec<Machine>, exploration_rate: f32) -> Self {
        
        Model {
            machines,
            estimates : vec![(0.0, 0); 10],
            exploration_rate,
            scores : vec![],
            name : format!("eps = {}", exploration_rate)
        }
    }


    fn select_machine(&self) -> usize{
        let mut best_index;

        // If any choice as not been taking yet, take it
        let not_yet_chosen_index = self.estimates
        .iter()
        .position(|&x| x.1 == 0);

        match not_yet_chosen_index {
            Some(x) => return x,
            None => ()
        }

        // Get the best index
        best_index = self.estimates
        .iter()
        .enumerate()
        .max_by(|x,y| compare(x.1.0 / x.1.1 as f32, y.1.0 / y.1.1 as f32))
        .unwrap().0;

        if self.exploration_rate == 0.0 {
            return best_index
        }

        if rand::thread_rng().gen::<f32>() < self.exploration_rate {
            let shift = rand::thread_rng().gen_range(0, self.machines.len() - 1);
            best_index = (best_index + shift) % self.machines.len();
        }
        // Explore
        
        
        best_index
    }

    fn update_model(&mut self, machine_index: usize) {
        let machine = &self.machines[machine_index];
        let value: f32 = machine.get_value();
        match self.scores.last() {
            Some(x) => self.scores.push(x+value),
            None => self.scores.push(value)
        }
        self.estimates[machine_index].0 += value;
        self.estimates[machine_index].1 += 1;
    }

    fn step(&mut self) {
        let machine_index = self.select_machine();
        self.update_model(machine_index);

    }

    fn get_estimates(&self) -> Vec<f32> {
        let mut result = vec![];
        for (total, times) in self.estimates.iter() {
            result.push(total / *times as f32);
        }
        result
    }
}

const COLORS: &[&RGBColor] = &[
    &GREEN, 
    &BLUE,
    &RED,
    &YELLOW,
    &CYAN,
    &MAGENTA,
    &BLACK,
];

fn draw_plot(models: Vec<Model>) {
    let path = "/home/loic/Documents/ai/rust_implementations/k-armed_bandits_problem/output_lines.png";
    let root_area = BitMapBackend::new(path, (1920, 1080))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let x_spec= ((N_STEPS as f32) * 1.1) as u32;
    let y_spec = (N_STEPS as f32 * DISTRIBUTION_RANGE) as u32;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Line Plot Demo", ("sans-serif", 40))
        .build_cartesian_2d(0..x_spec, 0..y_spec)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();
    for (index, model) in models.iter().enumerate() {
        ctx.draw_series(
            LineSeries::new((0..N_STEPS).map(|x| (x, model.scores[x as usize] as u32)), COLORS[index])
        ).unwrap()
        .label(&model.name)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], COLORS[index]));  
    }

    ctx.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
}

fn draw_histogram(scores: HashMap<String, u32>){
    let path = "/home/loic/Documents/ai/rust_implementations/k-armed_bandits_problem/output_histograms.png";
    let root_area = BitMapBackend::new(path, (1920, 1080))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let x_spec= scores.len() as u32;
    let y_spec = N_SAMPLE as u32;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Histo", ("sans-serif", 40))
        .build_cartesian_2d(0..x_spec, 0..y_spec)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // Draw the histogram
    ctx.draw_series((0..).zip(scores.values()).map(|(x, y)| {
        let mut bar = Rectangle::new([(x, 0), (x + 1, *y)], GREEN.filled());
        bar.set_margin(0, 0, 5, 5);
        bar
    }))
    .unwrap();

}

fn iterate(exploration_rates: &Vec<f32>) -> String{
    let machines : Vec<Machine> = (0..10).map(|_| Machine::new()).collect();
    
    let mut models = vec![];
    for exploration_rate in exploration_rates {
        let mut model = Model::new(&machines, *exploration_rate);
        for _ in 0..N_STEPS {
            model.step();
        }
        models.push(model);
    }


    models.iter().max_by_key(|m| *m.scores.last().unwrap() as u32).unwrap().exploration_rate.to_string()
    // println!("{} : {:?}", model.name, model.estimates);
    // println!("{:?}", model.get_estimates());
    
    // draw_plot(models);
}

fn main() {
    
    let exploration_rates = vec![0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5];
    let mut scores: HashMap<String, u32> = HashMap::new();
    for rate in exploration_rates.iter() {
        scores.insert(rate.to_string(), 0);
    }
    for _ in 0..N_SAMPLE {
        let winner: String = iterate(&exploration_rates);
        let value = scores.entry(winner).or_insert(0);
        *value += 1;
    }

    draw_histogram(scores);
    
}
