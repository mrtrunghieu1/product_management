/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package productmanagement;

/**
 *
 * @author Administrator
 */
public class productProperties {
    private String id;
    private String name;
    private String product_code; 
    private String category;
    private int retail_price; // Giá bán lẻ
    private int entry_price;
    private String describe;
    private int product_number;
    private int product_volume;
    
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getProduct_code() {
        return product_code;
    }

    public void setProduct_code(String product_code) {
        this.product_code = product_code;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public int getRetail_price() {
        return retail_price;
    }

    public void setRetail_price(int retail_price) {
        this.retail_price = retail_price;
    }

    public int getEntry_price() {
        return entry_price;
    }

    public void setEntry_price(int entry_price) {
        this.entry_price = entry_price;
    }

    public String getDescribe() {
        return describe;
    }

    public void setDescribe(String describe) {
        this.describe = describe;
    }

    public int getProduct_number() {
        return product_number;
    }

    public void setProduct_number(int product_number) {
        this.product_number = product_number;
    }

    public int getProduct_volume() {
        return product_volume;
    }

    public void setProduct_volume(int product_volume) {
        this.product_volume = product_volume;
    }
    
}
