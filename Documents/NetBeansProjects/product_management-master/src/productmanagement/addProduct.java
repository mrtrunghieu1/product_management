/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package productmanagement;

import java.sql.Connection;
import javax.swing.DefaultComboBoxModel;

/**
 *
 * @author Administrator
 */
public class addProduct extends javax.swing.JFrame {

    /**
     * Creates new form addProduct
     */
    private MyDBLib db = new MyDBLib();
    public Connection conn;
    
    public addProduct() {
        initComponents();
        choose_category();
    }

    private void choose_category(){
        DefaultComboBoxModel model = new DefaultComboBoxModel();
        model.addElement("");
        model.addElement("Phụ kiện thời trang nam");
        model.addElement("Thời trang nam");
        model.addElement("Thời trang nữ");
        model.addElement("Balo-Ví-Túi xách");
        choose_category.setModel(model);
    }
//    private void getting_info_product(){
//        productProperties base_product = new productProperties();
//        base_product.getId()
//    }
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jLabel4 = new javax.swing.JLabel();
        jPasswordField1 = new javax.swing.JPasswordField();
        jTextField6 = new javax.swing.JTextField();
        jProgressBar1 = new javax.swing.JProgressBar();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        product_name = new javax.swing.JTextField();
        product_ID = new javax.swing.JTextField();
        jLabel5 = new javax.swing.JLabel();
        jLabel6 = new javax.swing.JLabel();
        choose_category = new javax.swing.JComboBox<>();
        jLabel7 = new javax.swing.JLabel();
        jLabel8 = new javax.swing.JLabel();
        retail_price = new javax.swing.JTextField();
        jLabel10 = new javax.swing.JLabel();
        entry_price = new javax.swing.JTextField();
        jLabel12 = new javax.swing.JLabel();
        jLabel13 = new javax.swing.JLabel();
        jLabel14 = new javax.swing.JLabel();
        jLabel15 = new javax.swing.JLabel();
        jLabel16 = new javax.swing.JLabel();
        jTextField7 = new javax.swing.JTextField();
        jLabel17 = new javax.swing.JLabel();
        jPanel1 = new javax.swing.JPanel();
        jLabel9 = new javax.swing.JLabel();
        jLabel11 = new javax.swing.JLabel();
        product_number = new javax.swing.JTextField();
        jLabel18 = new javax.swing.JLabel();
        jTextField9 = new javax.swing.JTextField();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        notify_name = new javax.swing.JLabel();
        notify_names = new javax.swing.JLabel();
        dialog_name = new javax.swing.JLabel();
        dialog_ID = new javax.swing.JLabel();
        dialog_category = new javax.swing.JLabel();
        dialog_retail_price = new javax.swing.JLabel();
        dialog_entry_price = new javax.swing.JLabel();
        dialog_number = new javax.swing.JLabel();

        jLabel4.setText("jLabel4");

        jPasswordField1.setText("jPasswordField1");

        jTextField6.setText("jTextField6");

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setBackground(new java.awt.Color(255, 255, 255));

        jLabel2.setFont(new java.awt.Font("Tahoma", 1, 18)); // NOI18N
        jLabel2.setForeground(new java.awt.Color(0, 0, 204));
        jLabel2.setText("SẢN PHẨM MỚI");

        jLabel3.setText("   SẢN PHẨM MỚI");
        jLabel3.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 255)));

        jLabel1.setForeground(new java.awt.Color(0, 0, 102));
        jLabel1.setText("Tên sản phẩm(*)");

        product_name.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                product_nameMouseClicked(evt);
            }
        });
        product_name.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                product_nameActionPerformed(evt);
            }
        });

        product_ID.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                product_IDMouseClicked(evt);
            }
        });
        product_ID.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                product_IDActionPerformed(evt);
            }
        });

        jLabel5.setForeground(new java.awt.Color(0, 0, 102));
        jLabel5.setText("Mã sản phẩm(*)");

        jLabel6.setForeground(new java.awt.Color(0, 0, 102));
        jLabel6.setText("Danh mục sản phẩm");

        choose_category.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Item 1", "Item 2", "Item 3", "Item 4" }));
        choose_category.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                choose_categoryMouseClicked(evt);
            }
        });
        choose_category.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                choose_categoryActionPerformed(evt);
            }
        });

        jLabel7.setText("   GÍA SẢN PHẨM ");
        jLabel7.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 255)));

        jLabel8.setForeground(new java.awt.Color(0, 0, 102));
        jLabel8.setText("Gía bán lẻ ");

        retail_price.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                retail_priceMouseClicked(evt);
            }
        });
        retail_price.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                retail_priceActionPerformed(evt);
            }
        });

        jLabel10.setForeground(new java.awt.Color(0, 0, 102));
        jLabel10.setText("Giá nhập");

        entry_price.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                entry_priceMouseClicked(evt);
            }
        });
        entry_price.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                entry_priceActionPerformed(evt);
            }
        });

        jLabel12.setForeground(new java.awt.Color(0, 0, 153));
        jLabel12.setText("VND");
        jLabel12.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 204)));

        jLabel13.setForeground(new java.awt.Color(0, 0, 153));
        jLabel13.setText("VND");
        jLabel13.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 204)));

        jLabel14.setText("   MÔ TẢ");
        jLabel14.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 255)));

        jLabel15.setForeground(new java.awt.Color(0, 0, 102));
        jLabel15.setText("Hình ảnh ");

        jLabel16.setForeground(new java.awt.Color(0, 0, 102));
        jLabel16.setText("Mô tả");

        jLabel17.setText("   KHO HÀNG");
        jLabel17.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 255)));

        jPanel1.setBackground(new java.awt.Color(255, 255, 255));

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 296, Short.MAX_VALUE)
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 100, Short.MAX_VALUE)
        );

        jLabel9.setForeground(new java.awt.Color(0, 0, 102));
        jLabel9.setText("Đơn vị");

        jLabel11.setForeground(new java.awt.Color(0, 0, 102));
        jLabel11.setText("Số lượng");

        product_number.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                product_numberMouseClicked(evt);
            }
        });

        jLabel18.setForeground(new java.awt.Color(0, 0, 102));
        jLabel18.setText("SP");
        jLabel18.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(204, 204, 204)));

        jTextField9.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField9ActionPerformed(evt);
            }
        });

        jButton1.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        jButton1.setForeground(new java.awt.Color(255, 0, 0));
        jButton1.setText("Thêm sản phẩm");
        jButton1.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jButton1MouseClicked(evt);
            }
        });
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        jButton2.setForeground(new java.awt.Color(255, 0, 0));
        jButton2.setText("Hoàn tác");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        notify_names.setForeground(new java.awt.Color(255, 0, 0));

        dialog_name.setForeground(new java.awt.Color(255, 0, 0));

        dialog_ID.setForeground(new java.awt.Color(255, 0, 0));

        dialog_category.setForeground(new java.awt.Color(255, 0, 0));

        dialog_retail_price.setForeground(new java.awt.Color(255, 0, 0));

        dialog_entry_price.setForeground(new java.awt.Color(255, 0, 0));

        dialog_number.setForeground(new java.awt.Color(255, 0, 0));

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 217, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(195, 195, 195))
            .addComponent(jLabel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jLabel14, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jLabel17, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jLabel7, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGap(0, 0, Short.MAX_VALUE)
                        .addComponent(jLabel16)
                        .addGap(283, 283, 283))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addComponent(product_ID)
                                    .addComponent(product_name)
                                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                                                .addComponent(jLabel1)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(dialog_name, javax.swing.GroupLayout.PREFERRED_SIZE, 227, javax.swing.GroupLayout.PREFERRED_SIZE))
                                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                                                .addComponent(jLabel5)
                                                .addGap(18, 18, 18)
                                                .addComponent(dialog_ID, javax.swing.GroupLayout.PREFERRED_SIZE, 219, javax.swing.GroupLayout.PREFERRED_SIZE)))
                                        .addGap(0, 0, Short.MAX_VALUE)))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(notify_names)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(notify_name)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(choose_category, javax.swing.GroupLayout.PREFERRED_SIZE, 300, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(jLabel6)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(dialog_category, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(retail_price, javax.swing.GroupLayout.PREFERRED_SIZE, 273, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(jLabel8, javax.swing.GroupLayout.PREFERRED_SIZE, 62, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(dialog_retail_price, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(jLabel12)
                                .addGap(36, 36, 36)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(jLabel10)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(dialog_entry_price, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                    .addComponent(entry_price, javax.swing.GroupLayout.PREFERRED_SIZE, 273, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(jLabel13))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addGap(0, 0, Short.MAX_VALUE)
                                .addComponent(jTextField7, javax.swing.GroupLayout.PREFERRED_SIZE, 300, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(jLabel15)
                                    .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addGap(0, 0, Short.MAX_VALUE)))
                        .addContainerGap())
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel9)
                            .addComponent(jTextField9, javax.swing.GroupLayout.PREFERRED_SIZE, 300, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jLabel11)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(dialog_number, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                            .addComponent(product_number, javax.swing.GroupLayout.PREFERRED_SIZE, 277, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jLabel18)
                        .addContainerGap())))
            .addGroup(layout.createSequentialGroup()
                .addGap(96, 96, 96)
                .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 140, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 140, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(97, 97, 97))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel3, javax.swing.GroupLayout.PREFERRED_SIZE, 28, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel1)
                    .addComponent(jLabel6)
                    .addComponent(notify_name)
                    .addComponent(notify_names)
                    .addComponent(dialog_name)
                    .addComponent(dialog_category))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(product_name, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(choose_category))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel5)
                    .addComponent(dialog_ID))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(product_ID, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(10, 10, 10)
                .addComponent(jLabel7, javax.swing.GroupLayout.PREFERRED_SIZE, 28, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(7, 7, 7)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel8)
                    .addComponent(jLabel10)
                    .addComponent(dialog_retail_price)
                    .addComponent(dialog_entry_price))
                .addGap(2, 2, 2)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(retail_price, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel12)
                    .addComponent(entry_price, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel13))
                .addGap(18, 18, 18)
                .addComponent(jLabel14, javax.swing.GroupLayout.PREFERRED_SIZE, 28, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel15)
                    .addComponent(jLabel16))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jTextField7, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(jLabel17, javax.swing.GroupLayout.PREFERRED_SIZE, 28, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(jLabel11)
                        .addComponent(dialog_number))
                    .addComponent(jLabel9))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(product_number, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel18)
                    .addComponent(jTextField9, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(15, 15, 15)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void choose_categoryActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_choose_categoryActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_choose_categoryActionPerformed

    private void product_IDActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_product_IDActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_product_IDActionPerformed

    private void jTextField9ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField9ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jTextField9ActionPerformed

    private void product_nameActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_product_nameActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_product_nameActionPerformed

    private void jButton1MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jButton1MouseClicked
//        productProperties base_product = new productProperties();
////        base_product.setName((product_name.getText()=="")?"":product_name.getText());
//        base_product.setName((product_name.getText()));
////        System.out.println("-----"+base_product.getName());
////        if (base_product.getName() == ""){
////            notify_names.setText("ban chua dien cho nay");   
////        }
//        base_product.setId(product_ID.getText());
//        base_product.setCategory(choose_category.getItemAt(choose_category.getSelectedIndex()));
//        base_product.setRetail_price(Integer.parseInt(retail_price.getText()));
////        if (base_product.getRetail_price()== 0){
////            notify_names.setText("ban chua dien cho nay");
////        }
//        base_product.setEntry_price(Integer.parseInt(entry_price.getText()));
//        base_product.setProduct_number(Integer.parseInt(product_number.getText()));
////        System.out.println(base_product.getRetail_price());  
//        
//        db.add_info_product_db(base_product);
        
    }//GEN-LAST:event_jButton1MouseClicked

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        if (product_name.getText().trim().isEmpty() && product_ID.getText().trim().isEmpty() &&
                choose_category.getItemAt(choose_category.getSelectedIndex()).trim().isEmpty() &&
                retail_price.getText().trim().isEmpty() && entry_price.getText().trim().isEmpty()&&
                product_number.getText().trim().isEmpty())
        {
            dialog_name.setText("Bạn nhập thiếu tên");
            dialog_ID.setText("Bạn nhập thiếu ID");
            dialog_category.setText("Hãy chọn danh mục");
            dialog_retail_price.setText("Bạn nhập thiếu giá bán");
            dialog_entry_price.setText("Bạn nhập thiếu giá nhập vào");
            dialog_number.setText("Bạn chưa nhập số lượng");
        }
        else if(product_name.getText().trim().isEmpty())
        {
            dialog_name.setText("Bạn nhập thiếu tên");
        }
        else if(product_ID.getText().trim().isEmpty())
        {
            dialog_ID.setText("Bạn nhập thiếu ID");
        }
        else if (choose_category.getItemAt(choose_category.getSelectedIndex()).trim().isEmpty())
        {
            dialog_category.setText("Hãy chọn danh mục");
        }
        else if (retail_price.getText().trim().isEmpty())
        {
            dialog_retail_price.setText("Bạn nhập thiếu giá bán");
        }
        else if (entry_price.getText().trim().isEmpty())
        {
            dialog_entry_price.setText("Bạn nhập thiếu giá nhập vào");
        }
        else if (product_number.getText().trim().isEmpty())
        {   
            dialog_number.setText("Bạn chưa nhập số lượng");
        }
        else
        {
        db.dbConnect("root", "h02111998");
        
        productProperties base_product = new productProperties();
//      base_product.setName((product_name.getText()=="")?"":product_name.getText());
        base_product.setName((product_name.getText()));
//        System.out.println("-----"+base_product.getName());
//        if (base_product.getName() == ""){
//            notify_names.setText("ban chua dien cho nay");   
//        }
        base_product.setId(product_ID.getText());
        base_product.setCategory(choose_category.getItemAt(choose_category.getSelectedIndex()));
        base_product.setRetail_price(Integer.parseInt(retail_price.getText()));
//        if (base_product.getRetail_price()== 0){
//            notify_names.setText("ban chua dien cho nay");
//        }
        base_product.setEntry_price(Integer.parseInt(entry_price.getText()));
        base_product.setProduct_number(Integer.parseInt(product_number.getText()));
//        System.out.println(base_product.getRetail_price());  
        
        db.add_info_product_db(base_product);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void retail_priceActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_retail_priceActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_retail_priceActionPerformed

    private void entry_priceActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_entry_priceActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_entry_priceActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jButton2ActionPerformed

    private void product_nameMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_product_nameMouseClicked
        dialog_name.setText("");
    }//GEN-LAST:event_product_nameMouseClicked

    private void product_IDMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_product_IDMouseClicked
        dialog_ID.setText("");
    }//GEN-LAST:event_product_IDMouseClicked

    private void choose_categoryMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_choose_categoryMouseClicked
        dialog_category.setText("");
    }//GEN-LAST:event_choose_categoryMouseClicked

    private void retail_priceMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_retail_priceMouseClicked
        dialog_retail_price.setText("");
    }//GEN-LAST:event_retail_priceMouseClicked

    private void entry_priceMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_entry_priceMouseClicked
        dialog_entry_price.setText("");
    }//GEN-LAST:event_entry_priceMouseClicked

    private void product_numberMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_product_numberMouseClicked
        dialog_number.setText("");
    }//GEN-LAST:event_product_numberMouseClicked

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(addProduct.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(addProduct.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(addProduct.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(addProduct.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new addProduct().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JComboBox<String> choose_category;
    private javax.swing.JLabel dialog_ID;
    private javax.swing.JLabel dialog_category;
    private javax.swing.JLabel dialog_entry_price;
    private javax.swing.JLabel dialog_name;
    private javax.swing.JLabel dialog_number;
    private javax.swing.JLabel dialog_retail_price;
    private javax.swing.JTextField entry_price;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel10;
    private javax.swing.JLabel jLabel11;
    private javax.swing.JLabel jLabel12;
    private javax.swing.JLabel jLabel13;
    private javax.swing.JLabel jLabel14;
    private javax.swing.JLabel jLabel15;
    private javax.swing.JLabel jLabel16;
    private javax.swing.JLabel jLabel17;
    private javax.swing.JLabel jLabel18;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPasswordField jPasswordField1;
    private javax.swing.JProgressBar jProgressBar1;
    private javax.swing.JTextField jTextField6;
    private javax.swing.JTextField jTextField7;
    private javax.swing.JTextField jTextField9;
    private javax.swing.JLabel notify_name;
    private javax.swing.JLabel notify_names;
    private javax.swing.JTextField product_ID;
    private javax.swing.JTextField product_name;
    private javax.swing.JTextField product_number;
    private javax.swing.JTextField retail_price;
    // End of variables declaration//GEN-END:variables
}
