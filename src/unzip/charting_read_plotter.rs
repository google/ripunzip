// Copyright 2023 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::unzip::seekable_http_reader::ReadPlotter;
use crate::unzip::seekable_http_reader::ReadType;
use plotters::prelude::*;

const TRACE_PNG: &str = "read_pattern.png";

#[derive(Default)]
pub struct ChartingReadPlotter {
    total_length: u64,
    data: Vec<(u64, u64, ReadType)>,
}

impl ReadType {
    fn style(&self) -> ShapeStyle {
        match self {
            ReadType::FromCache => GREEN,
            ReadType::FromCacheDeferred => CYAN,
            ReadType::FromCacheAfterDirectAfterRewind => RED,
            ReadType::FromCacheAfterDirectAfterSkipOver => BLUE,
            ReadType::FromCacheAfterDirect => MAGENTA,
            ReadType::IntoCacheDuringFastForward => BLACK,
        }
        .filled()
    }
}

impl ReadPlotter for ChartingReadPlotter {
    fn set_len(&mut self, total_length: u64) {
        self.total_length = total_length;
    }
    fn plot_read(&mut self, start: u64, len: usize, read_type: ReadType) {
        if !matches!(read_type, ReadType::FromCache) {
            // too noisy
            let end = start + len as u64;
            self.data.push((start, end, read_type));
        }
    }
}

impl Drop for ChartingReadPlotter {
    fn drop(&mut self) {
        self.draw()
            .unwrap_or_else(|error| log::error!("Unable to save trace plot because {:?}", error));
    }
}

impl ChartingReadPlotter {
    fn draw(&self) -> Result<(), Box<dyn std::error::Error>> {
        let drawing_area = BitMapBackend::new(TRACE_PNG, (4096, 3072)).into_drawing_area();
        drawing_area.fill(&WHITE)?;

        let num_reads = self.data.len() as u64;

        let mut chart = ChartBuilder::on(&drawing_area)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .caption("Read pattern", ("sans-serif", 30.0).into_font())
            .build_cartesian_2d(0u64..self.total_length, 0u64..num_reads)?;

        chart.configure_mesh().light_line_style(WHITE).draw()?;

        chart.draw_series(self.data.iter().enumerate().map(
            |(counter, (start, end, read_type))| {
                Rectangle::new(
                    [(*start, counter as u64), (*end, counter as u64 + 1)],
                    read_type.style(),
                )
            },
        ))?;

        log::info!("Trace has been saved to {}", TRACE_PNG);
        Ok(())
    }
}
