//! Comprehensive tests for NumRS2 Financial Functions
//!
//! This test suite covers all financial functions including present value (PV),
//! future value (FV), payment (PMT), rate, number of periods (NPER),
//! net present value (NPV), and internal rate of return (IRR).

use approx::assert_relative_eq;
use numrs2::prelude::*;

#[test]
fn test_pv_basic_functionality() {
    // Test basic present value calculation
    let result = pv(0.05, 10.0, 100.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, -772.1734, epsilon = 1e-4);
}

#[test]
fn test_pv_with_future_value() {
    // Test present value with future value
    let result = pv(0.05, 10.0, 100.0, 1000.0, 0).unwrap();
    assert_relative_eq!(result, -1386.087, epsilon = 1e-3);
}

#[test]
fn test_pv_beginning_of_period() {
    // Test present value with payments at beginning of period
    let result = pv(0.05, 10.0, 100.0, 0.0, 1).unwrap();
    assert_relative_eq!(result, -810.782, epsilon = 1e-3);
}

#[test]
fn test_pv_zero_rate() {
    // Test present value with zero interest rate
    let result = pv(0.0, 10.0, 100.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, -1000.0, epsilon = 1e-9);
}

#[test]
fn test_pv_array() {
    let rates = Array::from_vec(vec![0.05, 0.06]);
    let npers = Array::from_vec(vec![10.0, 15.0]);
    let pmts = Array::from_vec(vec![100.0, 200.0]);
    let fvs = Array::from_vec(vec![0.0, 0.0]);

    let result = pv_array(&rates, &npers, &pmts, &fvs, 0).unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert_relative_eq!(values[0], -772.1734, epsilon = 1e-4);
    assert_relative_eq!(values[1], -1942.449798, epsilon = 1e-4);
}

#[test]
fn test_fv_basic_lump_sum() {
    // Test basic future value calculation for lump sum
    let result = fv(0.05, 10.0, 0.0, -1000.0, 0).unwrap();
    assert_relative_eq!(result, 1628.8946, epsilon = 1e-4);
}

#[test]
fn test_fv_annuity() {
    // Test future value of annuity
    let result = fv(0.05, 10.0, -100.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, 1257.7893, epsilon = 1e-4);
}

#[test]
fn test_fv_combined() {
    // Test future value with both present value and payments
    let result = fv(0.05, 10.0, -100.0, -1000.0, 0).unwrap();
    assert_relative_eq!(result, 2886.6839, epsilon = 1e-4);
}

#[test]
fn test_fv_beginning_of_period() {
    // Test future value with payments at beginning of period
    let result = fv(0.05, 10.0, -100.0, 0.0, 1).unwrap();
    assert_relative_eq!(result, 1320.6787, epsilon = 1e-4);
}

#[test]
fn test_fv_zero_rate() {
    // Test future value with zero interest rate
    let result = fv(0.0, 10.0, -100.0, -1000.0, 0).unwrap();
    assert_relative_eq!(result, 2000.0, epsilon = 1e-9);
}

#[test]
fn test_fv_array() {
    let rates = Array::from_vec(vec![0.05, 0.06]);
    let npers = Array::from_vec(vec![10.0, 15.0]);
    let pmts = Array::from_vec(vec![0.0, 0.0]);
    let pvs = Array::from_vec(vec![-1000.0, -2000.0]);

    let result = fv_array(&rates, &npers, &pmts, &pvs, 0).unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert_relative_eq!(values[0], 1628.8946, epsilon = 1e-4);
    assert_relative_eq!(values[1], 4793.116386, epsilon = 1e-4);
}

#[test]
fn test_pmt_basic_loan() {
    // Test basic loan payment calculation
    let monthly_rate = 0.05 / 12.0;
    let months = 5.0 * 12.0;
    let result = pmt(monthly_rate, months, 10000.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, -188.7107, epsilon = 2e-3);
}

#[test]
fn test_pmt_with_future_value() {
    // Test payment calculation with target future value
    let result = pmt(0.05, 10.0, 0.0, 10000.0, 0).unwrap();
    assert_relative_eq!(result, -795.04, epsilon = 1e-2);
}

#[test]
fn test_pmt_beginning_of_period() {
    // Test payment with payments at beginning of period
    let monthly_rate = 0.05 / 12.0;
    let months = 5.0 * 12.0;
    let result = pmt(monthly_rate, months, 10000.0, 0.0, 1).unwrap();
    assert_relative_eq!(result, -187.93, epsilon = 1e-2);
}

#[test]
fn test_pmt_zero_rate() {
    // Test payment with zero interest rate
    let result = pmt(0.0, 10.0, 1000.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, -100.0, epsilon = 1e-9);
}

#[test]
fn test_pmt_savings() {
    // Test payment for savings goal (negative PV, positive FV)
    let result = pmt(0.05, 10.0, 0.0, 10000.0, 0).unwrap();
    assert_relative_eq!(result, -795.04, epsilon = 1e-2);
}

#[test]
fn test_pmt_array() {
    let rates = Array::from_vec(vec![0.05 / 12.0, 0.06 / 12.0]);
    let npers = Array::from_vec(vec![60.0, 72.0]);
    let pvs = Array::from_vec(vec![10000.0, 15000.0]);
    let fvs = Array::from_vec(vec![0.0, 0.0]);

    let result = pmt_array(&rates, &npers, &pvs, &fvs, 0).unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert_relative_eq!(values[0], -188.7107, epsilon = 2e-3);
    assert_relative_eq!(values[1], -248.59, epsilon = 1e-2);
}

#[test]
fn test_rate_basic_loan() {
    // Test interest rate calculation for a known loan
    let monthly_rate = 0.05 / 12.0;
    let result = rate(
        60.0,
        -188.71,
        10000.0,
        0.0,
        0,
        Some(0.1),
        Some(1e-6),
        Some(100),
    )
    .unwrap();
    assert_relative_eq!(result, monthly_rate, epsilon = 1e-4);
}

#[test]
fn test_rate_simple_case() {
    // Simple case: find rate for doubling money in 10 periods with no payments
    let result = rate(
        10.0,
        0.0,
        -1000.0,
        2000.0,
        0,
        Some(0.1),
        Some(1e-6),
        Some(100),
    )
    .unwrap();
    let expected = 2.0_f64.powf(1.0 / 10.0) - 1.0; // ~7.18%
    assert_relative_eq!(result, expected, epsilon = 1e-6);
}

#[test]
fn test_rate_annuity() {
    // Test rate calculation for an annuity
    let result = rate(
        10.0,
        -100.0,
        772.17,
        0.0,
        0,
        Some(0.1),
        Some(1e-6),
        Some(100),
    )
    .unwrap();
    assert_relative_eq!(result, 0.05, epsilon = 1e-3);
}

#[test]
fn test_rate_zero_payment() {
    // Test rate with zero payment (simple compound interest)
    let result = rate(
        5.0,
        0.0,
        -1000.0,
        1276.28,
        0,
        Some(0.1),
        Some(1e-6),
        Some(100),
    )
    .unwrap();
    assert_relative_eq!(result, 0.05, epsilon = 1e-4);
}

#[test]
fn test_rate_array() {
    let npers = Array::from_vec(vec![10.0, 20.0]);
    let pmts = Array::from_vec(vec![0.0, 0.0]);
    let pvs = Array::from_vec(vec![-1000.0, -2000.0]);
    let fvs = Array::from_vec(vec![1628.89, 6536.00]);

    let result = rate_array(
        &npers,
        &pmts,
        &pvs,
        &fvs,
        0,
        Some(0.1),
        Some(1e-6),
        Some(100),
    )
    .unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert_relative_eq!(values[0], 0.05, epsilon = 1e-4);
    assert_relative_eq!(values[1], 0.06, epsilon = 1e-3);
}

#[test]
fn test_nper_basic_loan() {
    // Test number of periods for a loan payment
    let monthly_rate = 0.05 / 12.0;
    let result = nper(monthly_rate, -188.71, 10000.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, 60.0, epsilon = 1e-2);
}

#[test]
fn test_nper_savings_goal() {
    // Test number of periods to reach a savings goal
    let result = nper(0.05, -100.0, 0.0, 1257.79, 0).unwrap();
    assert_relative_eq!(result, 10.0, epsilon = 1e-2);
}

#[test]
fn test_nper_zero_payment() {
    // Test number of periods with no payment (simple compound interest)
    let result = nper(0.05, 0.0, -1000.0, 1628.89, 0).unwrap();
    assert_relative_eq!(result, 10.0, epsilon = 1e-2);
}

#[test]
fn test_nper_zero_rate() {
    // Test number of periods with zero interest rate
    let result = nper(0.0, -100.0, 1000.0, 0.0, 0).unwrap();
    assert_relative_eq!(result, 10.0, epsilon = 1e-9);
}

#[test]
fn test_nper_beginning_of_period() {
    // Test number of periods with payments at beginning
    let monthly_rate = 0.05 / 12.0;
    let result = nper(monthly_rate, -188.71, 10000.0, 0.0, 1).unwrap();
    // Should be slightly less than 60 months
    assert!(result < 60.0 && result > 58.0);
}

#[test]
fn test_nper_with_future_value() {
    // Test number of periods with both present and future values
    let result = nper(0.05, -200.0, 1000.0, 5000.0, 0).unwrap();
    assert!(result > 0.0); // Should be positive
}

#[test]
fn test_nper_array() {
    let rates = Array::from_vec(vec![0.05 / 12.0, 0.06 / 12.0]);
    let pmts = Array::from_vec(vec![-188.71, -250.0]);
    let pvs = Array::from_vec(vec![10000.0, 12000.0]);
    let fvs = Array::from_vec(vec![0.0, 0.0]);

    let result = nper_array(&rates, &pmts, &pvs, &fvs, 0).unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert_relative_eq!(values[0], 60.0, epsilon = 1e-2);
    assert!(values[1] > 0.0); // Should be positive
}

#[test]
fn test_npv_basic() {
    // Test basic NPV calculation
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0, 600.0]);
    let result = npv(0.1, &cash_flows).unwrap();
    assert_relative_eq!(result, 388.771259, epsilon = 1e-5);
}

#[test]
fn test_npv_zero_rate() {
    // Test NPV with zero discount rate
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
    let result = npv(0.0, &cash_flows).unwrap();
    assert_relative_eq!(result, 200.0, epsilon = 1e-9);
}

#[test]
fn test_npv_negative_rate() {
    // Test NPV with negative discount rate
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0]);
    let result = npv(-0.05, &cash_flows).unwrap();
    // With negative rate, future cash flows are discounted at negative rate
    // Expected: -1000 + 300/0.95 + 400/0.95^2 = -240.997
    assert_relative_eq!(result, -240.997230, epsilon = 1e-5);
}

#[test]
fn test_npv_single_cash_flow() {
    // Test NPV with only initial investment
    let cash_flows = Array::from_vec(vec![-1000.0]);
    let result = npv(0.1, &cash_flows).unwrap();
    assert_relative_eq!(result, -1000.0, epsilon = 1e-9);
}

#[test]
fn test_npv_positive_initial() {
    // Test NPV with positive initial cash flow
    let cash_flows = Array::from_vec(vec![1000.0, -300.0, -400.0, -500.0]);
    let result = npv(0.1, &cash_flows).unwrap();
    // Expected: 1000 - 300/1.1 - 400/1.1^2 - 500/1.1^3 = 21.037
    assert_relative_eq!(result, 21.036814, epsilon = 1e-5);
}

#[test]
fn test_npv_rates() {
    let rates = Array::from_vec(vec![0.05, 0.10, 0.15]);
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
    let result = npv_rates(&rates, &cash_flows).unwrap();
    assert_eq!(result.shape(), vec![3]);

    let values = result.to_vec();
    // Higher discount rates should give lower NPVs
    assert!(values[0] > values[1]);
    assert!(values[1] > values[2]);
}

#[test]
fn test_npv_multiple_series() {
    let cash_flows = Array::from_vec(vec![
        -1000.0, 300.0, 400.0, 500.0, // Project 1
        -1200.0, 400.0, 500.0, 600.0, // Project 2
    ])
    .reshape(&[2, 4]);
    let result = npv_multiple_series(0.1, &cash_flows).unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert_relative_eq!(values[0], -21.036814, epsilon = 1e-5); // Project 1 NPV
    assert_relative_eq!(values[1], 27.648385, epsilon = 1e-5); // Project 2 NPV
}

#[test]
fn test_irr_basic() {
    // Test basic IRR calculation
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0, 600.0]);
    let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
    assert_relative_eq!(result, 0.248883, epsilon = 1e-5);
}

#[test]
fn test_irr_simple_case() {
    // Simple case: invest $100, get back $110 next period
    let cash_flows = Array::from_vec(vec![-100.0, 110.0]);
    let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
    assert_relative_eq!(result, 0.1, epsilon = 1e-6);
}

#[test]
fn test_irr_break_even() {
    // Break-even case: IRR should be 0
    let cash_flows = Array::from_vec(vec![-100.0, 50.0, 50.0]);
    let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
    assert_relative_eq!(result, 0.0, epsilon = 1e-6);
}

#[test]
fn test_irr_high_return() {
    // High return case
    let cash_flows = Array::from_vec(vec![-100.0, 200.0]);
    let result = irr(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
    assert_relative_eq!(result, 1.0, epsilon = 1e-6); // 100% return
}

#[test]
fn test_irr_multiple_series() {
    let cash_flows = Array::from_vec(vec![
        -1000.0, 300.0, 400.0, 500.0, // Project 1
        -100.0, 110.0, 0.0, 0.0, // Project 2 (simple case)
    ])
    .reshape(&[2, 4]);
    let result = irr_multiple_series(&cash_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
    assert_eq!(result.shape(), vec![2]);

    let values = result.to_vec();
    assert!(values[0] > 0.0); // Project 1 should have positive IRR
    assert_relative_eq!(values[1], 0.1, epsilon = 1e-2); // Project 2 should be ~10%
}

#[test]
fn test_mirr_basic() {
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0, 400.0, 500.0]);
    let result = mirr(&cash_flows, 0.10, 0.12).unwrap();
    assert!(result > 0.0 && result < 1.0); // Should be a reasonable rate
}

#[test]
fn test_financial_error_cases() {
    // Test with empty array
    let empty_flows = Array::from_vec(Vec::<f64>::new());
    assert!(npv(0.1, &empty_flows).is_err());
    assert!(irr(&empty_flows, Some(0.1), Some(1e-6), Some(100)).is_err());

    // Test with NaN rate
    let cash_flows = Array::from_vec(vec![-1000.0, 300.0]);
    assert!(npv(f64::NAN, &cash_flows).is_err());

    // Test with infinite rate
    assert!(npv(f64::INFINITY, &cash_flows).is_err());

    // Test IRR with all positive cash flows
    let all_positive = Array::from_vec(vec![100.0, 200.0, 300.0]);
    assert!(irr(&all_positive, Some(0.1), Some(1e-6), Some(100)).is_err());

    // Test IRR with all negative cash flows
    let all_negative = Array::from_vec(vec![-100.0, -200.0, -300.0]);
    assert!(irr(&all_negative, Some(0.1), Some(1e-6), Some(100)).is_err());

    // Test NPER error cases
    assert!(nper(0.0, 0.0, 1000.0, 1000.0, 0).is_err());
}

#[test]
fn test_real_world_financial_scenarios() {
    // Scenario 1: 30-year mortgage
    let monthly_rate = 0.04 / 12.0; // 4% annual, monthly compounding
    let months = 30.0 * 12.0; // 30 years
    let loan_amount = 300000.0;

    let monthly_payment: f64 = pmt(monthly_rate, months, loan_amount, 0.0, 0).unwrap();
    assert!(monthly_payment < 0.0); // Payment should be negative (outflow)
    assert!(monthly_payment.abs() > 1000.0 && monthly_payment.abs() < 2000.0); // Reasonable range

    // Verify we can calculate back to original loan amount
    let calculated_pv = pv(monthly_rate, months, monthly_payment, 0.0, 0).unwrap();
    assert_relative_eq!(calculated_pv.abs(), loan_amount, epsilon = 1.0);

    // Scenario 2: Retirement savings
    let annual_rate = 0.07; // 7% annual return
    let years = 30.0; // 30 years to retirement
    let annual_contribution = -12000.0; // $12,000 per year (negative = payment)

    let retirement_value = fv(annual_rate, years, annual_contribution, 0.0, 0).unwrap();
    assert!(retirement_value > 1000000.0); // Should be over $1M

    // Scenario 3: Project evaluation
    let project_flows = Array::from_vec(vec![-50000.0, 15000.0, 20000.0, 25000.0, 30000.0]);
    let discount_rate = 0.08;

    let project_npv = npv(discount_rate, &project_flows).unwrap();
    assert!(project_npv > 0.0); // Project should be profitable

    let project_irr = irr(&project_flows, Some(0.1), Some(1e-6), Some(100)).unwrap();
    assert!(project_irr > discount_rate); // IRR should exceed discount rate

    // Scenario 4: Bond valuation
    let face_value = 1000.0;
    let coupon_rate = 0.05;
    let market_rate = 0.04;
    let years_to_maturity = 10.0;
    let annual_coupon = face_value * coupon_rate;

    let bond_price = pv(
        market_rate,
        years_to_maturity,
        -annual_coupon,
        -face_value,
        0,
    )
    .unwrap();
    assert!(bond_price > face_value); // Bond should trade at premium when coupon > market rate
}

#[test]
fn test_financial_functions_consistency() {
    // Test that the financial functions are consistent with each other
    let interest_rate = 0.06;
    let num_periods = 20.0;
    let payment = -500.0;
    let future_value = 0.0;

    // Calculate PV, then use it to verify other functions
    let calculated_pv = pv(interest_rate, num_periods, payment, future_value, 0).unwrap();

    // Verify FV calculation is consistent
    let calculated_fv = fv(interest_rate, num_periods, payment, calculated_pv, 0).unwrap();
    assert_relative_eq!(calculated_fv, future_value, epsilon = 1e-6);

    // Verify PMT calculation is consistent
    let calculated_pmt = pmt(interest_rate, num_periods, calculated_pv, future_value, 0).unwrap();
    assert_relative_eq!(calculated_pmt, payment, epsilon = 1e-6);

    // Verify NPER calculation is consistent
    let calculated_nper = nper(interest_rate, payment, calculated_pv, future_value, 0).unwrap();
    assert_relative_eq!(calculated_nper, num_periods, epsilon = 1e-6);

    // Verify RATE calculation is consistent (this one is harder due to numerical methods)
    let calculated_rate = rate(
        num_periods,
        payment,
        calculated_pv,
        future_value,
        0,
        Some(0.1),
        Some(1e-6),
        Some(100),
    )
    .unwrap();
    assert_relative_eq!(calculated_rate, interest_rate, epsilon = 1e-4);
}
